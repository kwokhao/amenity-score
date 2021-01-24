###
# Amenity score script v0.3.1 JAN-24-2021
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
gpd = pyimport("geopandas")
ctx = pyimport("contextily")

# common file addresses
git = "amenity-score/"
make_data_path = git * "make_data/"
data_path = git * "data/"

# import subzone path
subzonePath = git *
    "/data/master-plan-2014-subzone-boundary-no-sea/" *
    "master-plan-2014-subzone-boundary-no-sea-shp/" *
    "MP14_SUBZONE_NO_SEA_PL.shp"

"Imports the relevant data frames."
function loadData(; data_path=data_path)
    # import relevant data frames
    dh = load(data_path * "hawker-centres/hawkersCleaned.csv") |> DataFrame
    ds = load(data_path * "supermarkets/supermarketsCleaned.csv") |> DataFrame
    dm = load(data_path * "MRTStationCoords.csv") |> DataFrame
    # explicitly label `block` as String, else throws error
    df = CSV.File(data_path * "Apr012020_FlatsMerged2015-.csv",
                     types=Dict(:block => String)) |> DataFrame
    dD = load(make_data_path * "cleanedHDBDemographics.csv") |> DataFrame
    dD = @> dD @select(:postal_code, :fracOld, :fracYoung) unique
    
    # rename postal code, merge with demographic data, drop missing entries
    rename!(df, :POSTALCODE => :postal_code)
    df = @> df leftjoin(dD, on=:postal_code)
    dropmissing!(df, disallowmissing=true)
    
    # rescale prices to reasonable numbers
    df[!, :p] = df.resale_price / 1e5
    
    # focus on 3-room, 4-room and 5-room flats
    df = @> df @where(:flat_type .∈ [["3 ROOM", "4 ROOM", "5 ROOM"]])
    
    # find opening dates for hawker centres
    compl = [length(x) >= 7 ? tryparse(Int64, x[end-3:end]) :
             1965 for x in dh.EST_ORIGINAL_COMPLETION_DATE]
    dh[!, :yr] = [isnothing(x) ? 1965 : x for x in compl]
    
    
    # Dictionary mapping year to hawker centres active in that year
    hDict = Dict{Int64,DataFrame}(
        yr => @> dh @where(:yr .<= yr)
        for yr in levels(year.(df.month)))

    df, dh, ds, dm, dD, hDict
end

##
# 1b. Define useful structs
# --
#
# We define three structs to hold useful information:

# - ~inputStruct~ contains the raw amenity data frames (hawker CSV, supermarket
#   CSV, etc.) to construct ~distStruct~ and ~dataStruct~.
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

# struct of input data frames
@with_kw struct inputStruct
    dh::DataFrame  # hawker data frame
    ds::DataFrame  # supermarket data frame
    dm::DataFrame  # MRT data frame
    hDict::Dict  # dictionary of hawkers by year
end

# struct of pairwise distances
struct distStruct  # (i, j): distance of flat i from shop j
    Hawkers::Array{Float64, 2}  # hawkers
    Supermarkets::Array{Float64, 2}  # supermarkets
    MRTs::Array{Float64, 2}  # MRT stations
end

# construct relevant struct of columns in df
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
    scoreFood = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.Hawkers))
    scoreGroceries = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.Supermarkets))
    scoreMRT = log.(componentAmenityScore(1.0, κ[1], pairwiseDist.MRTs))

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
function splitTrainTest(df, frac=0.8; bootstrap=false)
    N = nrow(df)
    if bootstrap
        seq = rand(MersenneTwister(Threads.threadid() + abs(rand(Int))), 1:N, N)
        trainInd = seq[1:floor(Int, N * frac)]
        testInd = seq[floor(Int, N * frac) + 1:end]
    else
        seq = shuffle(1:N)
        trainInd = seq[1:floor(Int, N * frac)]
        testInd = seq[floor(Int, N * frac) + 1:end]
    end

    train = df[trainInd, :]
    test = df[testInd, :]
    train, test
end


"Given training and testing sets, computes `pairwiseDist` and `data`"
function prepTrainTest(train::DataFrame, test::DataFrame, t::Date=Date(2019,1,1);
                       inputStruct)
    
    @unpack hDict, ds, dm = inputStruct

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
    M = zeros(size(train, 1), size(dm, 1))  # MRT distances

    HTest = zeros(size(test, 1), size(hDict[yr], 1))  # hawker distances
    STest = zeros(size(test, 1), size(ds, 1))  # supermarket distances
    MTest = zeros(size(test, 1), size(dm, 1))  # MRT distances

    populateDistance!(H, train, hDict[yr])
    populateDistance!(S, train, ds)
    populateDistance!(M, train, dm)
    populateDistance!(HTest, test, hDict[yr])
    populateDistance!(STest, test, ds)
    populateDistance!(MTest, test, dm)

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
function computeRegression(train::DataFrame, test::DataFrame, t::Date=Date(2019,1,1);
                           inputStruct, verbose=false)

    data, dataTest, pairwiseDist, pairwiseDistTest = prepTrainTest(train, test, t,
                                                                   inputStruct=inputStruct)

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


###
# 4a. EXTENSION -- Rolling window analysis
# --
#
# Train over 2 months, then predict over next month, then slide window over
# 1 month.
###

"Splits data into rolling window, with start specified start date"
function splitTTWindow(df, t::Date; n_training_months=2, n_testing_months=1, bootstrap=false)

    train = @> df @where((:month .>= t) .& (:month .< t + Month(n_training_months)))
    test = @> df @where((:month .>= t + Month(n_training_months)) .&
                        (:month .< t + Month(n_training_months + n_testing_months)))

    if bootstrap
        train = train[rand(1:nrow(train), nrow(train)), :]
        test = test[rand(1:nrow(test), nrow(test)), :]
    end

    train, test
end


"Computes windowed regression and reports evolution of coefficients over time.
Optionally bootstraps coefficients for each run."
function predictAmenityScore(df, inputStruct; training_start_date=Date(2015, 1),
                             n_training_months=2, n_testing_months=1, S=12, bootstrap=false)
    
    # gets valid testing/training months
    monthRange = training_start_date:Month(1):(
            (df.month |> maximum) - Month(n_training_months + n_testing_months - 1))

    resDict = Dict{Symbol, Array{Float64, N} where N}(
        :κ => zeros(length(monthRange)),
        :β => zeros(length(monthRange), 14),
        :R2 => zeros(length(monthRange)),
        :pHat => Vector{Float64}[],
        :xHat => Array{Float64, 2}[]
    )

    # initialize regression results
    for i=tqdm(1:length(monthRange))
        train, test = splitTTWindow(df, monthRange[i])
        _, κ, β, predPrices, XTest, R2Test, _ =
            computeRegression(train, test, monthRange[i],
                              inputStruct=inputStruct,
                              verbose=true)
        resDict[:κ][i] = κ
        resDict[:β][i, :] = β
        resDict[:R2][i] = R2Test
        resDict[:pHat] = vcat(resDict[:pHat], predPrices)
        resDict[:xHat] = isempty(resDict[:xHat]) ? XTest : vcat(resDict[:xHat], XTest)
    end

    # optionally allow for bootstrap runs to visualize uncertainty in parameter estimates
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
                train, test = splitTTWindow(df, monthRange[i], bootstrap=true)
                _, κ, β, predPrices, _, R2Test, _ = computeRegression(
                    train, test, monthRange[i], verbose=true)
                bsDict[s][:κ][i] = κ
                bsDict[s][:β][i, :] = β
                bsDict[s][:R2][i] = R2Test
                bsDict[s][:pHat] = vcat(bsDict[s][:pHat], predPrices)
            end
        end

        return resDict, bsDict, monthRange
    end

    resDict, monthRange
end


"Filters and augments data frame rows with regression output. Assumes # of
testing months is 1 for now."
function augmentDf(df, resDict, monthRange; n_training_months=2)
    @assert n_training_months < 11 "# training months above 11 not supported in
    output CSV generation"

    # form a correspondence between months and runs of the algorithm
    testMonthRange = monthRange .+ Month(n_training_months)
   
    dfP = @where(df, :month .∈ [testMonthRange]) |> copy
    dfP[!, :t] = [findfirst(isequal.(mth, testMonthRange)) for mth in dfP.month]

    # extract predicted prices and amenity scores
    dfP[!, :pHat] = resDict[:pHat]
    dfP[!, :sHawker] = resDict[:xHat][:, 6]
    dfP[!, :sSuper] = resDict[:xHat][:, 9]
    dfP[!, :sMRT] = resDict[:xHat][:, 12]

    # extract regression parameter values
    dfP[!, :kappa] = [resDict[:κ][tt] for tt in dfP.t]
    dfP[!, :bFloorArea] = [resDict[:β][tt, 2] for tt in dfP.t]
    dfP[!, :bRemLease] = [resDict[:β][tt, 3] for tt in dfP.t]
    dfP[!, :b4Room] = [resDict[:β][tt, 4] for tt in dfP.t]
    dfP[!, :b5Room] = [resDict[:β][tt, 5] for tt in dfP.t]
    dfP[!, :aHawker] = [resDict[:β][dfP.t[i], 6] + resDict[:β][dfP.t[i], 7] * dfP.fracOld[i] +
                     resDict[:β][dfP.t[i], 8] * dfP.fracYoung[i] for i=1:size(dfP, 1)]
    dfP[!, :aSuper] = [resDict[:β][dfP.t[i], 9] + resDict[:β][dfP.t[i], 10] * dfP.fracOld[i] +
                    resDict[:β][dfP.t[i], 11] * dfP.fracYoung[i] for i=1:size(dfP, 1)]
    dfP[!, :aMRT] = [resDict[:β][dfP.t[i], 12] + resDict[:β][dfP.t[i], 13] * dfP.fracOld[i] +
                  resDict[:β][dfP.t[i], 14] * dfP.fracYoung[i] for i=1:size(dfP, 1)]
    dfP
end

"Generates an output CSV of amenity scores and parameters by resale flat.
Assumes # of testing months is 1 for now."
function genOutputCSV(df, resDict, monthRange; n_training_months=2,
                      output_file_name=make_data_path * "amenityscores.csv")
    
    @assert n_training_months < 11 "# training months above 11 not supported in
    output CSV generation"

    dfP = augmentDf(df, resDict, monthRange, n_training_months=n_training_months)
    CSV.write(output_file_name, dfP)
    
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
                     customlabel="prices (\$100k)",
                     output_directory=make_data_path)
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
        sns.scatterplot(x=model, y=data.p, style=data.flat_type,
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
        output_directory * "$(Date(Dates.now()))amenityscore_fitplot.png", dpi=300)
    plt.close()
    
    nothing
end



# regression 4-way plot
"Computes residuals vs. fitted; normal Q-Q; scale-location; and residuals vs.
leverage plots."
function regressionValidityPlots(p, fitted, X; output_directory=make_data_path)

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
        output_directory * "$(Date(Dates.now()))amenityscore_diagplot.png",
        dpi=300)
    plt.close()
    
    nothing
end


"Plots the kernel density plot of each amenity score type"
function plotAmenityScoreKDE(resDict; output_directory=make_data_path)
    hawkerscore = resDict[:xHat][:, 6]
    superscore = resDict[:xHat][:, 9]
    mrtscore = resDict[:xHat][:, 12]

    sns.kdeplot(hawkerscore, label="hawker", alpha=0.8)
    sns.kdeplot(superscore, label="supermarket", alpha=0.8)
    sns.kdeplot(mrtscore, label="MRT", alpha=0.8)
    plt.legend()
    plt.title("Kernel density plots of component amenity scores")
    plt.tight_layout()
    plt.savefig(output_directory * "amenityScoreKDE.png", dpi=300)
    plt.close()
    
    nothing
end


"Coerces component amenity scores by location into a pandas dataframe and plots
them on the Singapore map."
function plotAmenityScoreByLocation(resDict, dfP; output_directory=make_data_path)
    hawkerscore = resDict[:xHat][:, 6]
    superscore = resDict[:xHat][:, 9]
    mrtscore = resDict[:xHat][:, 12]
    sg = gpd.read_file(subzonePath).to_crs("epsg:4326")  # import subzone "backbone"
    gdfP = gpd.GeoDataFrame(Dict("LON" => dfP.LON,
                                 "LAT" => dfP.LAT,
                                 "p" => dfP.p,
                                 "size" => dfP.flat_type,
                                 "hawker" => hawkerscore,
                                 "super" => superscore,
                                 "mrt" => mrtscore),
                            geometry=gpd.points_from_xy(dfP.LON, dfP.LAT), crs="epsg:4326")

    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdfP.plot(column="hawker", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted Hawker Amenity Score")
    plt.tight_layout()
    plt.savefig(output_directory * "hawkerAmenityScore.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdfP.plot(column="super", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted Supermarket Amenity Score")
    plt.tight_layout()
    plt.savefig(output_directory * "superAmenityScore.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    sg.plot(ax=ax, alpha=0.0)  # to get correct dimensions for Singapore map
    gdfP.plot(column="mrt", legend=true, ax=ax,
              alpha=0.05, markersize=10)
    ctx.add_basemap(ax=ax, crs=4326)
    plt.suptitle("Predicted MRT Amenity Score")
    plt.tight_layout()
    plt.savefig(output_directory * "mrtAmenityScore.png", dpi=300)
    plt.close()

    nothing
end

"Plots evolution of component amenity scores over time, on aggregate"
function plotAmenityScoresOverTime(resDict, dfP; output_directory=make_data_path)
    mAS = [resDict[:xHat][:, 3+3i] for i=1:3]
    namesAS = ["HawkerScore", "SupermarketScore", "MRTScore"]

    dPlot = Dict(namesAS[i] => mAS[i] for i=1:3) |> DataFrame
    dPlot[!, :t] = dfP.month
    dPlot2 = @> dPlot groupby(:t) @combine(
        hM=mean(:HawkerScore), sM=mean(:SupermarketScore), mM=mean(:MRTScore),
        h10=quantile(:HawkerScore, 0.1), s10=quantile(:SupermarketScore, 0.1), m10=quantile(:MRTScore, 0.1),
        h90=quantile(:HawkerScore, 0.9), s90=quantile(:SupermarketScore, 0.9), m90=quantile(:MRTScore, 0.9))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i=1:length(mAS)
        ax[i].plot(1:48, dPlot2[:, i+1], label=namesAS[i])
        ax[i].fill_between(
            1:length(mAS[1]), dPlot2[:, i+4], dPlot2[:, i+7],
            alpha=0.5, color="C1", label="10th-90th Percentile")
        ax[i].set_xticks([0, 20, 40])
        ax[i].set_xticklabels(["2014/12", "2016/08", "2018/04"])
        ax[i].legend()
    end
    plt.suptitle("Evolution of mean component amenity scores of transacted flats over time")
    plt.savefig(output_directory * "amenityScoresOverTime.png", dpi=300)
end

"Plots amenity scores for 3 chosen flats over time"
function plotFlatAmenityScoreOverTime(dfP; output_directory=make_data_path)
    flatDict = Dict{Int64, DataFrame}()
    postcodes = [50034, 120507, 520283]
    amenities = [:sHawker, :sSuper, :sMRT]
    amenityNames = ["Hawker score", "Supermarket score", "MRT score"]
    for i=1:length(postcodes)
        flatDict[i] = @as x dfP[dfP.postal_code .== postcodes[i], :] unique(x, :month)
    end
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for i=1:length(postcodes), k=1:3
        ax[k, i].plot(1:size(flatDict[i], 1), flatDict[i][amenities[k]], label=amenityNames[k])
        ax[k, i].legend()
    end
    plt.suptitle("Amenity score plots over time for \n" *
                 "$(flatDict[1].block[1]) $(flatDict[1].street_name[1]), i.e. $(postcodes[1]) (left),\n" *
                 "$(flatDict[2].block[1]) $(flatDict[2].street_name[1]), i.e. $(postcodes[2]) (centre),\n" *
                 "and $(flatDict[3].block[1]) $(flatDict[3].street_name[1]), i.e. $(postcodes[3]) (right)")
    plt.tight_layout()
    plt.savefig(output_directory * "amenityScorePlotsFor3Flats.png", dpi=300)
    
end



"Plots evolution of amenity score weights over time"
function plotWeightsOverTime(resDict, dfP;
                             bsDict=nothing, bootstrapped=false,
                             output_directory=make_data_path)
    
    mASW = Vector{Matrix{Float64}}()
    namesASW = ["Hawker Weight", "Supermarket Weight", "MRT Weight"]
    
    # plot evolution of amenity score weights over time (bootstrapped)
    if bootstrapped & !isnothing(bsDict)
        mASB = [(Dict(s => bsDict[s][:β][:, 3+3i] for s in levels(keys(bsDict))) |> DataFrame |> Array)
               for i=1:3]
        mASO = [(Dict(s => bsDict[s][:β][:, 3+3i+1] for s in levels(keys(bsDict))) |> DataFrame |> Array)
                for i=1:3]
        mASY = [(Dict(s => bsDict[s][:β][:, 3+3i+2] for s in levels(keys(bsDict))) |> DataFrame |> Array)
                for i=1:3]
    
        # evaluate score weights at mean level of `fracOld` and `fracYoung`
        for i=1:3
            push!(mASW, mASB[i] .+ mASO[i] .* mean(dfP.fracYoung) .+ mASY[i] .* mean(dfP.fracYoung))
        end
    
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
    else
        # plot evolution of amenity score weights using main data
        vASB = [resDict[:β][:, 3+3i] for i=1:3]
        vASO = [resDict[:β][:, 3+3i+1] for i=1:3]
        vASY = [resDict[:β][:, 3+3i+2] for i=1:3]
        vASW = [vASB[i] .+ vASO[i] .* mean(dfP.fracYoung) .+ vASY[i] .* mean(dfP.fracYoung) for i=1:3]

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for i=1:length(vASW)
            ax[i].plot(1:length(vASW[1]), vASW[i], label=namesASW[i])
            ax[i].set_xticks([0, 20, 40])
            ax[i].set_xticklabels(["2014/12", "2016/08", "2018/04"])
            ax[i].legend()
        end
    end
    plt.suptitle("Evolution of amenity score weights α over time, \n" *
                 "evaluated at mean fractions of old and young in each HDB block")
    plt.savefig(output_directory * "$(Date(Dates.now()))ASWeights.png", dpi=300)
    
    nothing
end


"Main function generating plots. Assumes # of testing months is 1 for now."
function generatePlots(resDict, df, monthRange; bsDict=nothing, n_training_months=2,
                       output_directory=make_data_path)
    testMonthRange = monthRange .+ Month(n_training_months)
    dfP = @where(df, :month .∈ [testMonthRange]) |> copy
    
    plotFitPlot(resDict[:pHat], dfP, isDataFrame=true,
                output=true, kde=false, output_directory=output_directory)
    regressionValidityPlots(dfP.p, resDict[:pHat], resDict[:xHat],
                            output_directory=output_directory)
    plotAmenityScoreKDE(resDict, output_directory=output_directory)
    plotAmenityScoreByLocation(resDict, dfP, output_directory=output_directory)
    plotWeightsOverTime(resDict, dfP,
                        bsDict=bsDict, output_directory=output_directory)
end

###
# MAIN FUNCTION
###

function main(; output_directory=make_data_path)
    println("Loading data...")
    df, ds, dh, dm, dD, hDict = loadData()
    ip = inputStruct(dh=dh, ds=ds, dm=dm, hDict=hDict)
    println("Predicting amenity score...")
    resDict, monthRange = predictAmenityScore(
        df, ip, training_start_date=Date(2015, 1),
        n_training_months=2, n_testing_months=1, bootstrap=false)
    
    # output data
    println("Generating output CSV...")
    genOutputCSV(df, resDict, monthRange,
                 n_training_months=2, output_file_name=output_directory * "amenityscores.csv")
    # save plots
    println("Generating plots...")
    generatePlots(resDict, df, monthRange, n_training_months=2, output_directory=output_directory)
    println("Done!")

    nothing
end


###
# A. APPENDIX: TIME-SERIES VARIATION IN RESALE PRICES
###

"Plots time-series variation in resale prices by flat size, for 3-room, 4-room
and 5-room flats."
function plotTSVarResalePrices(df)
    dTS = @> df begin
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
