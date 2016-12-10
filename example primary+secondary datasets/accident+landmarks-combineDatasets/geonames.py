import configparser
import csv
import requests
import urllib
from sqlalchemy import create_engine
import pandas as pd

'''
geonames.py loads up data from geonames, using configuration in datasets.config
'''


def makeUrl(prefix, params):
    url = prefix + '?'
    count=0
    for p in params:
        if '=' in p:
            url += p.strip().split('=')[0] + '=' + p.strip().split('=')[1] + '&'
        else:
            url += p.strip()+'={'+str(count)+'}&'
            count+=1
    return url[:len(url)-1]

def fetch_landmarks(url, lat, long):
    landmarks = []
    url = url.format(lat, long)
    json_response = requests.get(url)
    response = json_response.json()
    for neighbor in response["geonames"]:
        landmark_distance,landmark_type, landmark_lat, landmark_long, landmark_geonameId, landmark_fcode = -1,-1, -1, -1, -1, -1
            
        if 'distance' in neighbor:  landmark_distance = neighbor["distance"]
        if 'fcl' in neighbor:       landmark_type = neighbor["fcl"]
        if 'lat' in neighbor:       landmark_lat = neighbor["lat"]
        if 'lng' in neighbor:       landmark_long = neighbor["lng"]
        if 'geonameId' in neighbor: landmark_geonameId = neighbor["geonameId"]
        if 'fcode' in neighbor:     landmark_fcode = neighbor["fcode"]
            
        landmarks.append({'lat':landmark_lat,'lng':landmark_long,'distance':landmark_distance, 'type':landmark_type,
                          'geonameId':landmark_geonameId, 'fcode':landmark_fcode  })
    return landmarks



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("datasets.config")  
    
    ## Form rest api query for landmarks
    ## it returns distance in kilometers
    url_prefix = config.get('LandmarkApi','ApiUrl')
    params = config.get('LandmarkApi','QueryParams')
    url = makeUrl(url_prefix, params.strip().split(','))
    print ( "Fetching data from following rest API:", url )
        
    ## Form postgres query to fetch accidents, and save data in primary file
    
    engine_url = config.get('Postgres','PythonEngineURL')
    engine = create_engine(engine_url.strip())
    
    postgres_query = config.get('Postgres','SqlQuery_0')
    df = pd.read_sql_query(postgres_query+" "+config.get('Postgres','topK'), con=engine)
    
    #df = df.dropna(axis=0)
    # change number of vehicles from an integer to bins
    df.numvehs=pd.cut(df.numvehs, [0,1,2,3,5,8,13,21,100], labels=["1","2","3","3-5","5-8","8-13","13-21","21-"])
    
    ## write out primary file
    fout = open(config.get('datasets','outPrimaryFile'),'w')
    fout.write("lat__float, long_float, acctype__cat, light__cat, numvehs__cat, pop_grp__cat, rdsurf__cat, rodwycls__cat," +
               "weather__cat, hour__float, drv_age__int, drv_sex__cat, vehtype__cat, vehyr__int, severity__cat\n")
    
        # Note:  need to add a negative ign before longitude recalled from HSIS dataset
    for id,caseno in enumerate(df['caseno']):
        data_row = str(df['lat'][id])+","+" -"+str(df['long'][id])+","+str(df['acctype'][id])+","+str(df['light'][id])+","+ \
                    str(df['numvehs'][id])+","+str(df['pop_grp'][id])+","+str(df['rdsurf'][id])+","+str(df['rodwycls'][id])+","+ \
                    str(df['weather'][id])+","+str(df['hour'][id])+","+str(df['drv_age'][id])+","+str(df['drv_sex'][id])+","+ \
                    str(df['vehtype'][id])+","+str(df['vehyr'][id])+","+str(df['severity'][id])
        
        if ( df['vehyr'][id] != 'nan' ):
            if ( ( float( df['vehyr'][id] ) > 1922 ) and ( float( df['vehyr'][id] ) < 2023 ) ):
                fout.write(data_row + "\n")
    
    fout.close()
    
    ## write out secondary file
    fout = open(config.get('datasets','outSecondaryFile'),'w')   #  creae new file, overwriting previous one
    fout.write("header row\n")
    fout.close()
    with open(config.get('datasets','outPrimaryFile'), 'r') as f:
        r = csv.reader(f, delimiter=',')
        next(r)
        rowID = 0
        for row in r:
            with open(config.get('datasets','outSecondaryFile'), 'a') as f2:
                landmarks = fetch_landmarks(url,row[0], row[1])
                rowID += 1
                data_row = str(rowID)+","
                data_row += str(row[0])+","+str(row[1])+","
                for lm in landmarks:
                    data_row += ( str(lm['type']).strip() + ":" +str(lm['lat']).strip() + ":" +str(lm['lng']).strip() \
                    + ":" +str(lm['distance']).strip() + ":" +str(lm['fcode']).strip() + ":" +str(lm['geonameId']).strip() + "," )
                f2.write(data_row[:len(data_row)-1]+"\n")
            

        



    
