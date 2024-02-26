# geoconstrained-Voronoi
A python module for building geographically mapped Voronoi cells, used in my case to build healthcare analysis of South Africa with healthsites.io and WorldPop datasets. This is the clean version of the code I did for <a href='https://medium.com/odi-research/open-data-science-to-be-proud-of-geographic-healthcare-analysis-with-healthsites-io-and-worldpop-e33ee98d8a81'>this piece of data science</a>.

## Index
* gcVoronoi.py is the module built to automate the process. It's not easy to explain the way the methods work but they do, so that's all that matters right?
* SA_v1.ipynb is the notebook developed for the above piece, which shows how to use the module alongside the method I did to integrate WorldPop data into the project. It takes a while to run, so be careful.
* results/hospitals_w_population contains a healthsites shapefile for South African hospitals, with an extra column for population in their corresponding Voronoi cell (see the notebook for context)
* results/cells_w_population contains a shapefile for the geoconstrained Voronoi cells of SA_v1.ipynb , with an extra column for population within them (see the notebook for context)
* results/interactive_choropleth.html is an interactive map built to showcase final results.

## Images
![image](https://github.com/NevadaM/geoconstrained-Voronoi/assets/100001600/358d93cb-8603-4ce6-9df4-c9c9388ca7ff)
![image](https://github.com/NevadaM/geoconstrained-Voronoi/assets/100001600/3a1d54c3-cd1f-4640-abb4-6ccd210e2c11)
![image](https://github.com/NevadaM/geoconstrained-Voronoi/assets/100001600/64691060-8855-4979-95c3-ae10a704a407)

## Contact
neil.majithia@live.co.uk 
