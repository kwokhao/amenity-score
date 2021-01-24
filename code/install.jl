using Pkg
package_list = ["StatsBase", "Statistics", "StatFiles",
                "CSV", "FileIO", "CSVFiles",
                "DataFrames", "DataFramesMeta",
                "Random", "Distributions",
                "Lazy", "Dates", "Optim", "ProgressBars",
                "Parameters", "JLD2", "PyCall", "PyPlot"]
for pkg in package_list
    Pkg.add(pkg)
end

