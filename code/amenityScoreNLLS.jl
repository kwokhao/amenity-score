###
# Amenity score NLLS v0.3.0 JAN-20-2021
# ---
#
# This is the (current) master file for the amenity score NLLS
# procedure.
###

###
# 1a. Import packages and useful files
# --
#
# We import cleaned data sets for our resale flat prices (`dMRaw`), associated
# demographic data (`dD`), as well as our amenity sources
# `supermarketsCleaned.csv`, `hawkersCleaned.csv` and `MRTStationCoords.csv`. We
# then sample from the HDB transaction data set if necessary, then merge HDB
# transactions with associated demographics.
###

using StatsBase, Statistics, LinearAlgebra
using StatFiles, CSV, FileIO, CSVFiles
using DataFrames, DataFramesMeta
using Random, Distributions
using Lazy: @>, @>>, @as
using Dates
using Optim
using ProgressBars
using Parameters
using JLD2: @load, @save

# import Python packages for plotting
using PyCall, PyPlot
sns = pyimport("seaborn"); sns.set()
pd = pyimport("pandas")
sm = pyimport("statsmodels.api")
np = pyimport("numpy")


# common file addresses
git = "/Users/kwokhao/GoogleDrive/Research/mrt/amenity-score/"

# import subzone path
subzonePath = git *
    "/data/master-plan-2014-subzone-boundary-no-sea/" *
    "master-plan-2014-subzone-boundary-no-sea-shp/" *
    "MP14_SUBZONE_NO_SEA_PL.shp"

# import relevant data frames
ds = load(string(git, "data/supermarkets/supermarketsCleaned.csv")) |> DataFrame
dh = load(string(git, "data/hawker-centres/hawkersCleaned.csv")) |> DataFrame
dMRT = load(string(git, "data/MRTStationCoords.csv")) |> DataFrame
# explicitly label `block` as String, else throws error
dM = CSV.File(git * "data/Apr012020_FlatsMerged2015-.csv",
                 types=Dict(:block => String)) |> DataFrame
dD = load(git * "make_data/cleanedHDBDemographics.csv") |> DataFrame
dD = @> dD @select(:postal_code, :fracOld, :fracYoung) unique

# rename postal code, merge with demographic data, drop missing entries
rename!(dM, :POSTALCODE => :postal_code)
dM = @> dM leftjoin(dD, on=:postal_code)
dropmissing!(dM, disallowmissing=true)

# rescale prices to reasonable numbers
dM[:p] = dM.resale_price / 1e5

# focus on 3-room, 4-room and 5-room flats
dM = @> dM @where(:flat_type .∈ [["3 ROOM", "4 ROOM", "5 ROOM"]])

# find opening dates for hawker centres
compl = [length(x) >= 7 ? tryparse(Int64, x[end-3:end]) :
         1965 for x in dh.EST_ORIGINAL_COMPLETION_DATE]
dh[:yr] = [isnothing(x) ? 1965 : x for x in compl]


# Dictionary mapping year to hawker centres active in that year
const hDict = Dict{Int64,DataFrame}(
    yr => @> dh @where(:yr .<= yr)
    for yr in levels(year.(dM.month)))

##
# 1b. Define useful structs
# --
#
# We define two structs to hold useful information:
# - ~distStruct~ contains distances from flat i to shop j for each amenity type k
# - ~dataStruct~ contains the prediction target ~p~, or resale price in S$100k,
#   and various regressors:
#   - ~floorArea~: in sqm
#   - ~remLease~: remaining lease in years
#   - ~fourRoom~: indicators for 4-room flats
#   - ~fiveRoom~: indicators for 5-room flats
#   - ~fracOld~: the fraction of households in a block of flats that is above 70,
#   - ~fracYoung~: the fraction of households in a block of flats that is below 10,
#   - ~year~: year in which transaction was made
##

# struct of pairwise distances
struct distStruct  # (i, j): distance of flat i from shop j
    H::Array{Float64, 2}  # hawkers
    S::Array{Float64, 2}  # supermarkets
    M::Array{Float64, 2}  # MRT stations
end

# construct relevant struct of columns in dM
@with_kw struct dataStruct
    p::Vector{Float64}  # vector of apt prices
    floorArea::Vector{Float64}  # vector of floor areas
    remLease::Vector{Float64}  # vector of remaining leases
    fourRoom::Vector{Float64}  # vector of indicators for 4-room flats
    fiveRoom::Vector{Float64}  # vector of indicators for 4-room flats
    fracOld::Vector{Float64}  # vector of fraction of old people
    fracYoung::Vector{Float64}  # vector of fraction of young people
    # year::Vector{Int64}  # years
end


###
# 2. COMPUTE PAIRWISE DISTANCES
# --
#
# Computes pairwise distances τ_{ijk} between flat i and shop j of type k, then
# stores these pairwise distances in a matrix. Currently, we are using three
# matrices:
#
# - `H` is a N by J_hawker matrix, with each row depicting the distance between a
#   specific flat and a hawker centre.
# - `S` is a N by J_supermarket matrix...
# - `M` is a N by J_MRTStation matrix...
###

# define haversine function
const R = 6378.0

"Takes in (latX, lngX, latY, lngY) and returns the
distance in km between them."
function haversine(s_lat::Float64, s_lng::Float64,
                   e_lat::Float64, e_lng::Float64)::Float64
    d = sind((e_lat - s_lat)/2)^2 + cosd(s_lat) * cosd(e_lat) * sind((e_lng - s_lng)/2)^2
    return 2 * R * asin(sqrt(d))
end

# precompute struct of realized pairwise distances
function populateDistance!(distMat::Array{Float64, 2}, d::DataFrame, dShops::DataFrame)
    Threads.@threads for i=1:size(distMat, 1)
        for j=1:size(distMat, 2)
            distMat[i, j] = haversine(d.LAT[i], d.LON[i], dShops.LAT[j], dShops.LON[j])
        end
    end
end

###
# 3. Define accessory functions to compute amenity score
# --
#
# Contains
# - `shopContrib`, measuring a shop's contribution to my amenity score: 1/exp(τ0 + κ * τ_ijk)
# - `componentAmenityScore`, which sums the `shopContrib`s
# - `linearLoss`, which computes the deviation between predicted prices and the
#    actual resale prices in sample
###

# "Takes in house and amenity instance and computes the contribution to my
# amenity score"
# function amenityInstanceContrib(τijk, τ0, κ, A=1.0)
#     A/exp(τ0 + κ*τijk)
# end


"Takes in one house and a vector of amenity instance coordinates and computes my
component amenity score"
function componentAmenityScore(τ0, κ, distMat)
    # scoreMat = amenityInstanceContrib.(distMat, τ0, κ)
    scoreMat = 1.0 ./ exp.(τ0 .+ κ .* distMat)
    return(dropdims(sum(scoreMat, dims=2), dims=2))
end


# The NLLS loss function with logged amenity scores. Avoids crazy nonlinearities
function linearLoss(κ, scoreFood, scoreGroceries, scoreMRT;
                    d, pairwiseDist, verbose=false)
    # make struct with current parameters;
    # τ0 still not Id; doesn't matter what `alpha` is
    # currentParams = ASParams(1.0, κ[1], ones(3)/3)

    # compute amenity scores
    scoreFood = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.H))
    scoreGroceries = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.S))
    scoreMRT = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.M))

    X = hcat(ones(length(d.p)), d.floorArea, d.remLease, d.fourRoom, d.fiveRoom,
             scoreFood, scoreFood .* d.fracOld, scoreFood .* d.fracYoung,
             scoreGroceries, scoreGroceries .* d.fracOld, scoreGroceries .* d.fracYoung,
             scoreMRT, scoreMRT .* d.fracOld, scoreMRT .* d.fracYoung)
             # years.y2015, years.y2016, years.y2017, years.y2018, years.y2019)  # undo time FE

    β = X \ d.p
    # β = (inv(X.T * X) * X.T) * d.p  # long-form regression equation

    !verbose && return(d.p - X * β)  # residuals if not finding beta
    return X, β
end


###
# 4. REGRESSION
# --
#
# Splits data into a testing (80%) and training (20%) set. Initializes memory
# for component amenity scores, then computes regression to find marginal travel
# cost κ. Given κ, finds the other regression parameters, then computes
# component amenity scores. Optionally computes bootstrapped parameters.
###

"Splits data into training and testing set, 80-20 for now"
function splitTrainTest(dM, frac=0.8; bootstrap=false)
    N = nrow(dM)
    if bootstrap
        seq = rand(MersenneTwister(Threads.threadid() + abs(rand(Int))), 1:N, N)
        trainInd = seq[1:floor(Int, N * frac)]
        testInd = seq[floor(Int, N * frac) + 1:end]
    else
        seq = shuffle(1:N)
        trainInd = seq[1:floor(Int, N * frac)]
        testInd = seq[floor(Int, N * frac) + 1:end]
    end

    train = dM[trainInd, :]
    test = dM[testInd, :]
    train, test
end


"Given training and testing sets, computes `pairwiseDist` and `data`"
function prepTrainTest(train::DataFrame, test::DataFrame, t::Date=Date(2019,1,1))

    yr = year(t)
        
    # extract data from dataframes
    data = dataStruct(
        p=train.p, floorArea=train.floor_area_sqm, remLease=train.remaining_lease,
        fourRoom=(train.flat_type .== "4 ROOM"),
        fiveRoom=(train.flat_type .== "5 ROOM"),
        fracOld=train.fracOld, fracYoung=train.fracYoung)
    dataTest = dataStruct(
        p=test.p, floorArea=test.floor_area_sqm, remLease=test.remaining_lease,
        fourRoom=(test.flat_type .== "4 ROOM"),
        fiveRoom=(test.flat_type .== "5 ROOM"),
        fracOld=test.fracOld, fracYoung=test.fracYoung)

    
    # compute pairwise distances for training and testing sets
    H = zeros(size(train, 1), size(hDict[yr], 1))  # hawker distances
    S = zeros(size(train, 1), size(ds, 1))  # supermarket distances
    M = zeros(size(train, 1), size(dMRT, 1))  # MRT distances

    HTest = zeros(size(test, 1), size(hDict[yr], 1))  # hawker distances
    STest = zeros(size(test, 1), size(ds, 1))  # supermarket distances
    MTest = zeros(size(test, 1), size(dMRT, 1))  # MRT distances
    
    populateDistance!(H, train, hDict[yr])
    populateDistance!(S, train, ds)
    populateDistance!(M, train, dMRT)
    populateDistance!(HTest, test, hDict[yr])
    populateDistance!(STest, test, ds)
    populateDistance!(MTest, test, dMRT)

    pairwiseDist = distStruct(H, S, M)
    pairwiseDistTest = distStruct(HTest, STest, MTest)

    # years
    # years = Dict(yr => data.year .== yr for yr in unique(data.year)) |> DataFrame
    # @> years names!(Symbol.(["y" * String(yr) for yr in names(years)]))
    
    # yearsTest = Dict(yr => dataTest.year .== yr for yr in unique(dataTest.year)) |> DataFrame
    # @> yearsTest names!(Symbol.(["y" * String(yr) for yr in names(yearsTest)]))

    data, dataTest, pairwiseDist, pairwiseDistTest #, years, yearsTest    
end


"Computes regression performance and coefficients given training and testing set
for data"
function computeRegression(train::DataFrame, test::DataFrame, t::Date=Date(2019,1,1); verbose=false)
    
    data, dataTest, pairwiseDist, pairwiseDistTest = prepTrainTest(train, test, t)
    
    # preallocate arrays
    scoreFood = similar(zeros(size(train, 1)))
    scoreGroceries = similar(scoreFood)
    scoreMRT = similar(scoreFood)
    scoreFoodTest = similar(zeros(size(test, 1)))
    scoreGroceriesTest = similar(scoreFoodTest)
    scoreMRTTest = similar(scoreFoodTest)

    # optimize `linearLoss`, now using native Julia package
    κ0 = [0.500]
    f(z) = mean(linearLoss(z, scoreFood, scoreGroceries, scoreMRT,
                           d=data, pairwiseDist=pairwiseDist) .^ 2)
    res = optimize(f, κ0, LBFGS())
    κ = res.minimizer[1]
    
    # compute realized (X, β) | θ, R^2 and rmse on training at testing sets
    X, β = linearLoss(
        κ, scoreFood, scoreGroceries, scoreMRT,
        d=data, pairwiseDist=pairwiseDist, verbose=true)

    if !verbose
        return(κ, β)
    else
        # R2 = 1 - sum((data.p - X*β) .^ 2)/ sum((data.p .- mean(data.p)) .^ 2)
        # rmse = sqrt(mean((data.p - X*β) .^ 2))
        
        XTest, _ = linearLoss(
            κ, scoreFoodTest, scoreGroceriesTest, scoreMRTTest,
            d=dataTest, pairwiseDist=pairwiseDistTest, verbose=true)

        predPrices = XTest * β
        R2Test = 1 - sum((dataTest.p - predPrices) .^ 2) /
            sum((dataTest.p .- mean(dataTest.p)) .^ 2)
        rmseTest = sqrt(mean((dataTest.p - predPrices) .^ 2))
        
    
        return(X, κ, β, predPrices, XTest, R2Test, rmseTest)
    end    
end

"Computes bootstrap distribution of parameters for STATIC MODEL for subsequent analysis."
function bootstrapParameters(S=10; dM=dM)
    κList = zeros(S)
    βList = zeros(S, 14)
    Threads.@threads for s=1:S
        @show s
        train, test = splitTrainTest(dM, bootstrap=true)
        κ, β = computeRegression(train, test)
        κList[s] = κ
        βList[s, :] = β
    end
    κList, βList
end

"Code to run old procedure"
function _runStaticRegression(dM)
    train, test = splitTrainTest(dM, bootstrap=false)
    κ, β = computeRegression(train, test, verbose=false)
    K, B = bootstrapParameters()
    nothing
end

###
# 4a. EXTENSION -- Rolling window analysis
# --
#
# Train over 2 months, then predict over next month, then slide window over
# 1 month.
###

"Splits data into rolling window, with start specified start date"
function splitTTWindow(dM, t::Date; bootstrap=false)

    train = @> dM @where((:month .>= t) .& (:month .< t + Month(2)))
    test = @> dM @where((:month .>= t + Month(2)) .& (:month .< t + Month(3)))

    if bootstrap
        train = train[rand(1:nrow(train), nrow(train)), :]
        test = test[rand(1:nrow(test), nrow(test)), :]
    end
    
    train, test
end


"Computes windowed regression and reports evolution of coefficients over time.
Optionally bootstraps coefficients for each run."
function computeWindowRegression(dM; S=12, bootstrap=false)
    # gets valid initialization months
    monthRange = (dM.month |> minimum):Month(1):((dM.month |> maximum) - Month(2))

    resDict = Dict{Symbol, Array{Float64, N} where N}(
        :κ => zeros(length(monthRange)),
        :β => zeros(length(monthRange), 14),
        :R2 => zeros(length(monthRange)),
        :pHat => Vector{Float64}[],
        :xHat => Array{Float64, 2}[]
    )
    
    # initialize regression results
    for i=tqdm(1:length(monthRange))
        train, test = splitTTWindow(dM, monthRange[i])
        _, κ, β, predPrices, XTest, R2Test, _ = computeRegression(train, test, monthRange[i], verbose=true)
        resDict[:κ][i] = κ
        resDict[:β][i, :] = β
        resDict[:R2][i] = R2Test
        resDict[:pHat] = vcat(resDict[:pHat], predPrices)
        resDict[:xHat] = isempty(resDict[:xHat]) ? XTest : vcat(resDict[:xHat], XTest)
    end

    if bootstrap
        bsDict = Dict{Int64, Dict{Symbol, Array{Float64, N} where N}}()
        Threads.@threads for s=1:S
            bsDict[s] = Dict{Symbol, Array{Float64, N} where N}(
                :κ => zeros(length(monthRange)),
                :β => zeros(length(monthRange), 14),
                :R2 => zeros(length(monthRange)),
                :pHat => Vector{Float64}[]
            )
            for i=1:length(monthRange)
                train, test = splitTTWindow(dM, monthRange[i], bootstrap=true)
                _, κ, β, predPrices, _, R2Test, _ = computeRegression(
                    train, test, monthRange[i], verbose=true)
                bsDict[s][:κ][i] = κ
                bsDict[s][:β][i, :] = β
                bsDict[s][:R2][i] = R2Test
                bsDict[s][:pHat] = vcat(bsDict[s][:pHat], predPrices)
            end
        end
        
        return resDict, bsDict
    end
    
    resDict
end


"Code to run sliding window procedure"
function _runWindowRegression(dM; load_archived=false)

    FN = git * "make_data/amenityscore.jld2"
    # load from archive
    if load_archived
        @load FN resDict bsDict
    else
        resDict, bsDict = computeWindowRegression(dM, bootstrap=true)
        # archive amenity score dictionaries
        @save FN resDict bsDict
    end


    # export amenity score + flat characteristics to CSV
    dMP[:pHat] = resDict[:pHat]
    dMP[:hawker] = resDict[:xHat][:, 6]
    dMP[:super] = resDict[:xHat][:, 9]
    dMP[:MRT] = resDict[:xHat][:, 12]
    CSV.write("/Users/kwokhao/Desktop/resalepriceswithAS.csv", dMP)
    
    # plot changes in marginal travel cost over time
    dK = Dict(s => bsDict[s][:κ] for s=1:12) |> DataFrame |> Array
    plt.plot(1:48, mean(dK, dims=2))
    plt.fill_between(
        1:48, quantile.(eachrow(dK), 0.1), quantile.(eachrow(dK), 0.9),
        alpha=0.5, color="C1")


    sns.scatterplot(resDict[:xHat][:, 6], dMP.p, style=dMP.flat_type,
                        hue=dMP.flat_type, alpha=0.5)
    nothing
end

###
# 5. POST-REGRESSION DIAGNOSTICS
# --
#
# Computes the fit plot, histograms of amenity scores by price, etc. 
###


"Plots pretty fit plot of model predicted prices against realized resale prices"
function plotFitPlot(model, data; isDataFrame=false, kde=false,
                     output=false, range_restr=false,
                     customlabel="prices (\$100k)")
    # overlay CCP distributions
    diagLineMax = max(maximum(model), maximum(isDataFrame ? data.p : data))
    incr = diagLineMax / 20
    β = hcat(ones(size(data, 1)), (isDataFrame ? data.p : data)) \ model

    x = (min(minimum(isDataFrame ? data.p : data), minimum(model)):incr:diagLineMax)
    y = β[1] .+ β[2] * x

    

    # compute binscatter
    # pData, pModel = binscatter(data[:], model[:])

    labelToPlot = customlabel
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    if kde
        # kernel density
        sns.kdeplot(model, label="Predicted $(labelToPlot)", ax=ax[1])
        sns.kdeplot((isDataFrame ? data.p : data),
                    label="Data $(labelToPlot)", ax=ax[1])
    else
        # histogram
        ax[1].hist(model, bins=x,
                   histtype="step", linestyle="dashed",
                   label="Predicted $(labelToPlot)")
        ax[1].hist((isDataFrame ? data.p : data), bins=x,
                   alpha=0.5, label="Data $(labelToPlot)")
    end
    
    ax[1].legend()

    # plot fit correlations
    if isDataFrame
        sns.scatterplot(model, data.p, style=data.flat_type,
                        hue=data.flat_type, alpha=0.2, ax=ax[2])
    else
        ax[2].scatter(data, model, alpha=0.05, s=10,
                      color="k", label="Model vs. data $(labelToPlot)")
    end
    
    # ax[2].scatter(pData.values, pModel.values, alpha=1.0,
    #               color="C2", marker="1", s=50,
    #               label="Binscatter: Model vs. Data")
    ax[2].plot(x, y, linestyle="dashed", label="Best fit line", color="k")
    ax[2].plot(x, x, linestyle="dotted", label="45° line", color="k", alpha=0.5)
    ax[2].legend()
    ax[2].set_xlabel("Data")
    ax[2].set_ylabel("Predicted")
    fig.suptitle("Histogram of " *
                 "model $(labelToPlot) vis-a-vis the data (left) \n" *
                 "and associated scatter plot (right)")
    plt.tight_layout()
    output && plt.savefig(
        git * "make_data/$(Date(Dates.now()))amenityscore_fitplot_" *
        """$(replace(customlabel, " " => "_")).png""", dpi=300)
    plt.show()
end



# regression 4-way plot
"Computes residuals vs. fitted; normal Q-Q; scale-location; and residuals vs.
leverage plots."
function regressionValidityPlots(p, fitted, X)
    
    # compute residuals
    e = p - fitted
    e_std = e ./ std(e)
    e_r_std = @as rr e_std abs.(rr) sqrt.(rr)
    
    # initialize plot figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. residuals vs. fitted
    ax[1, 1].scatter(fitted, e, alpha=0.25)
    ax[1, 1].plot(np.unique(fitted), np.poly1d(np.polyfit(fitted, e, 1))(np.unique(fitted)),
                  alpha=0.8, color="C1")
    ax[1, 1].set_xlabel("Fitted values")
    ax[1, 1].set_ylabel("Residuals")
    ax[1, 1].set_title("Residuals vs. fitted")

    # 2. quantile-quantile plot
    sm.qqplot(e, line="45", ax=ax[1, 2]);
    ax[1, 2].set_title("Normal Q-Q")
    
    # 3. scale-location plot
    ax[2, 1].scatter(fitted, e_r_std, alpha=0.25)
    ax[2, 1].plot(np.unique(fitted), np.poly1d(np.polyfit(fitted, e_r_std, 1))(np.unique(fitted)),
                  alpha=0.8, color="C1")
    ax[2, 1].set_xlabel("Fitted values")
    ax[2, 1].set_ylabel("Root standardized residuals")
    ax[2, 1].set_title("Scale-Location")

    # 4. residual-leverage plot. To avoid inverting a large matrix, sample 5000 entries
    usePerm = randperm(MersenneTwister(1234), size(X, 1))[1:5000]
    eS = e_std[usePerm]
    XS = X[usePerm, :]
    lev = XS*inv(XS'XS)*XS' |> diag

    ax[2, 2].scatter(lev, eS, alpha=0.25)
    ax[2, 2].plot(np.unique(lev), np.poly1d(np.polyfit(lev, eS, 1))(np.unique(lev)),
                  alpha=0.8, color="C1")
    ax[2, 2].set_xlabel("Leverage")
    ax[2, 2].set_ylabel("Standardized residuals")
    ax[2, 2].set_title("Residuals vs. leverage")
    
    plt.tight_layout()
    plt.savefig(
        git * "make_data/$(Date(Dates.now()))amenityscore_diagplot.png",
        dpi=300)
end

"Joint plots of two fields in an output data frame. Currently not in use."
function _plotJointPlotBySize(x=:hawkerscore, y=:mrtscore;
                             town="BUKIT PANJANG", dOut, output=false)
    jg = @as z dOut begin
        @where(z, :flat_type .∈ [["3 ROOM", "4 ROOM", "5 ROOM"]])
        @where(z, :town .== town)
        sns.jointplot(z[!, x], z[!, y], hue=z.flat_type,
                      height=6)
    end
    jg.set_axis_labels(String(x), String(y))
    plt.suptitle("`$(String(x))` and `$(String(y))` for resale flats in $(town) \n by flat size")
    plt.tight_layout()
    output && plt.savefig(
        git * "make_data/$(Date(Dates.now()))amenityscore_jointplot_" *
        "$(String(y))_on_$(String(x))_$(town).png", dpi=300)
end


"Plots the kernel density plot of each amenity score type"
function plotAmenityScoreKDE(resDict)
    hawkerscore = resDict[:xHat][:, 6]
    superscore = resDict[:xHat][:, 9]
    mrtscore = resDict[:xHat][:, 12]

    sns.kdeplot(hawkerscore, label="hawker", alpha=0.8)
    sns.kdeplot(superscore, label="supermarket", alpha=0.8)
    sns.kdeplot(mrtscore, label="MRT", alpha=0.8)
    plt.legend()
    plt.title("Kernel density plots of component amenity scores")
    plt.tight_layout()
    plt.savefig(git * "make_data/amenityScoreKDE.png", dpi=300)
end


"Coerces component amenity scores by location into a pandas dataframe and plots
them on the Singapore map."
function plotAmenityScoreByLocation(resDict, dMP)
    hawkerscore = resDict[:xHat][:, 6]
    superscore = resDict[:xHat][:, 9]
    mrtscore = resDict[:xHat][:, 12]
    gpd = pyimport("geopandas")
    ctx = pyimport("contextily")
    sg = gpd.read_file(subzonePath).to_crs("epsg:4326")  # import subzone "backbone"
    gdMP = gpd.GeoDataFrame(Dict("LON" => dMP.LON,
                                 "LAT" => dMP.LAT,
                                 "p" => dMP.p,
                                 "size" => dMP.flat_type,
                                 "hawker" => hawkerscore,
                                 "super" => superscore,
                                 "mrt" => mrtscore),
                            geometry=gpd.points_from_xy(dMP.LON, dMP.LAT), crs="epsg:4326")

    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdMP.plot(column="hawker", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted Hawker Amenity Score")
    plt.tight_layout()
    plt.savefig(git * "make_data/hawkerAmenityScore.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdMP.plot(column="super", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted Supermarket Amenity Score")
    plt.tight_layout()
    plt.savefig(git * "make_data/superAmenityScore.png", dpi=300)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdMP.plot(column="mrt", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted MRT Amenity Score")
    plt.tight_layout()
    plt.savefig(git * "make_data/mrtAmenityScore.png", dpi=300)
    
end

"Plots evolution of component amenity scores over time"
function plotAmenityScoresOverTime(resDict, dMP)
    mAS = [resDict[:xHat][:, 3+3i] for i=1:3]
    namesAS = ["HawkerScore", "SupermarketScore", "MRTScore"]
    
    dPlot = Dict(namesAS[i] => mAS[i] for i=1:3) |> DataFrame
    dPlot[:t] = dMP.month
    dPlot2 = @> dPlot groupby(:t) @combine(
        hM=mean(:HawkerScore), sM=mean(:SupermarketScore), mM=mean(:MRTScore),
        h10=quantile(:HawkerScore, 0.1), s10=quantile(:SupermarketScore, 0.1), m10=quantile(:MRTScore, 0.1),
        h90=quantile(:HawkerScore, 0.9), s90=quantile(:SupermarketScore, 0.9), m90=quantile(:MRTScore, 0.9))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i=1:length(mAS)
        ax[i].plot(1:48, dPlot2[:, i+1], label=namesAS[i])
        ax[i].fill_between(
            1:48, dPlot2[:, i+4], dPlot2[:, i+7],
            alpha=0.5, color="C1", label="10th-90th Percentile")
        ax[i].set_xticks([0, 20, 40])
        ax[i].set_xticklabels(["2014/12", "2016/08", "2018/04"])
        ax[i].legend()
    end
    plt.suptitle("Evolution of mean component amenity scores of transacted flats over time")
    plt.savefig(git * "make_data/amenityScoresOverTime.png", dpi=300)

    
end


"Plots evolution of amenity score weights over time"
function plotWeightsOverTime(bsDict, dMP)
    # plot evolution of amenity score weights over time: mean ("base"), frac old, frac young
    mASB = [(Dict(s => bsDict[s][:β][:, 3+3i] for s in levels(keys(bsDict))) |> DataFrame |> Array)
           for i=1:3]
    mASO = [(Dict(s => bsDict[s][:β][:, 3+3i+1] for s in levels(keys(bsDict))) |> DataFrame |> Array)
            for i=1:3]
    mASY = [(Dict(s => bsDict[s][:β][:, 3+3i+2] for s in levels(keys(bsDict))) |> DataFrame |> Array)
            for i=1:3]

    # evaluate score weights at mean level of `fracOld` and `fracYoung`
    mASW = Vector{Matrix{Float64}}()
    for i=1:3
        push!(mASW, mASB[i] .+ mASO[i] .* mean(dMP.fracYoung) .+ mASY[i] .* mean(dMP.fracYoung))
    end
    
    namesASW = ["Hawker Weight", "Supermarket Weight", "MRT Weight"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i=1:length(mASW)
        ax[i].plot(1:48, mean(mASW[i], dims=2), label=namesASW[i])
        ax[i].fill_between(
            1:48, quantile.(eachrow(mASW[i]), 0.1), quantile.(eachrow(mASW[i]), 0.9),
            alpha=0.5, color="C1")
        ax[i].set_xticks([0, 20, 40])
        ax[i].set_xticklabels(["2014/12", "2016/08", "2018/04"])
        ax[i].legend()
    end
    plt.suptitle("Evolution of amenity score weights over time, \n" *
                 "evaluated at mean fractions of old and young in each HDB block")
    plt.savefig(git * "make_data/ASWeightOverTime.png", dpi=300)

end

"Main function generating plots"
function _generatePlots(resDict, bsDict, dM)
    dMP = @where(dM, :month .>= Date(2015, 3)) |> copy
    plotFitPlot(resDict[:pHat], dMP, isDataFrame=true, output=true, kde=false)
    regressionValidityPlots(dMP.p, resDict[:pHat], resDict[:xHat])
    plotAmenityScoreKDE(resDict)
    plotAmenityScoreByLocation(resDict, dMP)
    plotWeightsOverTime(bsDict, dMP)
end

###
# MAIN FUNCTION
###

function main()
    resDict, bsDict = _runWindowRegression(dM, bootstrap=true)
end


###
# A. APPENDIX: TIME-SERIES VARIATION IN RESALE PRICES
###

"Plots time-series variation in resale prices by flat size, for 3-room, 4-room
and 5-room flats."
function plotTSVarResalePrices(dM)
    dTS = @> dM begin
        groupby([:month, :flat_type])
        combine(:p .=> [mean, z -> quantile(z, 0.1), z -> quantile(z, 0.9)] .=> [:p_mean, :p10, :p90])
    end
    
    gdTS = @> dTS begin
        @where(:flat_type .∈ [["3 ROOM", "4 ROOM", "5 ROOM"]])
        groupby(:flat_type)
    end
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))    
    for i=1:3
        grp = gdTS[i]
        ax[i].plot(1:length(grp.month), grp.p_mean, label="Mean price, $(grp.flat_type[1])")
        ax[i].fill_between(1:length(grp.month), grp.p10, grp.p90,
                           alpha=0.5, label="Symmetric 80% interval of prices")
        ax[i].set_xticks(collect(0:10:50))
        ax[i].set_xticklabels(["2014/12", "2015/10", "2016/08", "2017/06", "2018/04", "2019/02"])
        ax[i].set_xlabel("Month")
        ax[i].set_ylabel("Resale price (S\$100k)")
        ax[i].legend()
    end
    plt.suptitle("Time series of resale prices by flat size")
    plt.tight_layout()
    plt.savefig(git * "make_data/$(Date(Dates.now()))resaleprices_timeseries.png",
                dpi=300)
end
