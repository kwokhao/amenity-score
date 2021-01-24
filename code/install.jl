using Pkg
package_list = ["StatsBase", "Statistics", "StatFiles",
                "CSV", "FileIO", "CSVFiles",
                "DataFrames", "DataFramesMeta",
                "Random", "Distributions",
                "Lazy", "Dates", "Optim", "ProgressBars",
                "Parameters", "JLD2", "Conda", "PyCall", "PyPlot"]
for pkg in package_list
    Pkg.add(pkg)
end

import Conda
Conda.pip_interop(true)
py_package_list = ["numpy", "matplotlib", "seaborn", "pandas",
                   "statsmodels", "geopandas", "contextily", "descartes"]
for pkg in py_package_list
    Conda.pip("install", pkg)
end

