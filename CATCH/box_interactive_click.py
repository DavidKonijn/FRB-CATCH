import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob

import argparse

parser = argparse.ArgumentParser(description="""Interactive clicky-click...""")
parser.add_argument('-f','--frb',type=str,required=True)
parser.add_argument('-d','--dateobs',type=float,required=True)
args = parser.parse_args()

candidates = glob.glob("/data/hewitt/eclat/"+args.frb+"/"+args.dateobs+"/catch/*png")
candidate_burst_number = []
coords = []

#sort the candidates based on their integer burst_id as sort on string ids sort differently
# e.g. [1, 10, 100, 2, 200, 200] ->  [1, 2, 10, 20, 100, 200]
for i in range(len(candidates)):
    candidate_burst_number.append(int(candidates[i].split('/')[-1].split('_')[0][1:]))

candidates = [x for _, x in sorted(zip(candidate_burst_number, candidates))]

#create a interactive onclick event that registers coordinates
def onclick(event):
    #scale the x and y coordinates to be in mHz and timesteps
    ix, iy = (event.xdata-200)/570*12500, (event.ydata-90)/620*128
    print(event.xdata, event.ydata)
    global coords
    coords.append(ix)
    coords.append(iy)
    #close the image when clicked on either red/green area or more than 6 times.
    if len(coords) == 12 or ix < 1000:
        fig.canvas.mpl_disconnect(cid)
        plt.close()

df = pd.DataFrame(columns=['Real Burst Good Box', 'No Burst','Real Burst Wrong Box', 'Extra Burst', 'Both'])

#read in all the candidates
for i in range(10):
    img=mpimg.imread(candidates[i])
    fig = plt.figure(figsize=(20,30))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    imgplot = plt.imshow(img)
    plt.show(block=True)

    #check in which area was clicked

    if coords[0] < 1000:
        if coords[1] >= 64:
            #green area
            df.loc[i,'Real Burst Good Box'] = True
        if coords[1] < 64:
            #red area
            df.loc[i,'No Burst'] = True
    elif coords[0] > 11500:
        if coords[2] >11500:
            #both orange and yellow
            df.loc[i,'Both'] = str(coords[4:])
        elif coords[1] >= 64:
            #orange area
            df.loc[i,'Real Burst Wrong Box'] = str(coords[2:])
        elif coords[1] < 64:
            #yellow area
            df.loc[i,'Extra Burst'] = str(coords[2:])

    coords = []
    df.to_csv('selected_burst_truths.csv')
