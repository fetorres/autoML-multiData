In this example problem, we add landmark data to the analysis of HSIS(1) accident data.  For example, one might want to see whether including the presence of nearby schools or parks improves the predictive power of a classifier for accident severity.

<b>Primary file:</b>  primary.data.csv contains data for accidents, including the latitude and longitude.  (The information does not include personal information.)  The [Postgres] section in the file datasets.config lists the queries used to extract the desired accident data.  The output of the SQL queries was used to create the primary.data.csv file. <br>
<b>Auxiliary data:</b>  We use landmark data from the Geonames service.  This data is available over the internet through an API, so the relevant data needs to be downloaded and then transformed into the format that autoML uses.

<b>geonames.py:</b> gets the auxiliary data from a specified url, the primary data from a SQL database loaded with IL HSIS data, and produces the secondary file to be used by autoML.  The url and api parameters are specified in datasets.config.  This program transforms the auxiliary data into the format expected by autoML, which is described in the document "UsingDistanceFunctionToJoinDisparateDatasets.pdf" in the "example datasets" parent folder. 
<br> The Geonames database has an efficient API for finding landmarks close to any specified latitude and longitude, so we take advantage of that here.  For each accident in primary.data.csv, geonames.py finds landmarks within the "radius" parameter specified in the [LandmarkAPI] section of datasets.config, repeating the process for each accident.  A radius of 5 km was chosen with the idea that analyses could be performed over a range of smaller radii within autoML.

The output of geonames is the outSecondaryFile specified in datasets.config, "secondary.data" in the example here.

Finally, a header row is manually added to secondary.data to produce secondary.data.csv.

(1)  HSIS data provided by the Federal Highway Administration Highway Safety Information System program, as part of an Exploratory Advanced Research project being conducted by the Palo Alto Research Center.
