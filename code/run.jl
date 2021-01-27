include("amenityScoreScript.jl")
println("Welcome to the amenity score project. Either")
println("(1) Type `main()` to run the amenity score estimation with defaults; or")
println("""(2) Run `main()` with options. You may specify (any of) the following:
function main(; input_data_path=data_path, output_directory=make_data_path,
              training_start_date=Date(2015, 1),
              n_training_months=2, n_testing_months=1,
              hawker_subpath="hawker-centres/hawkersCleaned.csv",
              supermarket_subpath="supermarkets/supermarketsCleaned.csv",
              mrt_subpath="MRTStationCoords.csv",
              resale_subpath="Apr012020_FlatsMerged2015-.csv",
              demographics_subpath="cleanedHDBDemographics.csv")
""")
# main()
