import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pykalman import KalmanFilter
from xml.dom.minidom import getDOMImplementation


   
def output_gpx(data, output_filename):
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    data.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def read_gpx(file):
    parse_result = ET.parse(file)

    def element_to_data(elem):
        lat = float(elem.get('lat'))
        lon = float(elem.get('lon'))
        return lat, lon

    trkpt_elements = parse_result.iter('{http://www.topografix.com/GPX/1/0}trkpt')
    return pd.DataFrame(list(map(element_to_data, trkpt_elements)), columns=['lat', 'lon'])


def calculate_distance(df):
   # from https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/ adapted

    p = np.pi / 180

    first = -np.cos((df['lat2'] - df['lat']) * p) / 2

    second = np.cos(df['lat'] * p) * np.cos(df['lat2'] * p)

    third = (1 - np.cos((df['lon2'] - df['lon']) * p)) / 2

  
    a = 0.5 + first + second * third

   
    df['distance'] = 12742000 * np.arcsin(np.sqrt(a))

   
    return df['distance'].sum()



def shift(data):

    shifts = data.shift(periods=1)

    data['lat2'] = shifts['lat']

    data['lon2'] = shifts['lon']
    
    return data.drop(data.index[0])


def filter(data):
    istate = data.iloc[0]

    tcovariance = np.diag([10, 10]) ** 2
#between 15-20
    obcovariance = np.diag([16, 16]) ** 2

    transitionM = [[1, 0], [0, 1]]
    
    kf = KalmanFilter(
        initial_state_mean=istate,

        transition_covariance=tcovariance,

        observation_covariance=obcovariance,

        transition_matrices=transitionM
    )
    
    smoothed_points, _ = kf.smooth(data)
    return pd.DataFrame({'lat': smoothed_points[:, 0], 'lon': smoothed_points[:, 1]})





def main():
    
    input_file = "walk1.gpx"
    data = read_gpx(input_file)

  
    shifted = shift(data.copy())

   
    unfiltered_distance = calculate_distance(shifted)

   
    smooth = filter(data)

   
    shiftedsm = shift(smooth.copy())

  
    filtered_distance = calculate_distance(shiftedsm)

    
    print('Unfiltered distance: %0.2f meters' % unfiltered_distance)

    print('Filtered distance: %0.2f meters' % filtered_distance)


    output_gpx(smooth, 'out.gpx')


if __name__ == '__main__':
    main()