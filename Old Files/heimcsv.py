#heimcsv
# writes the Heimdall candidates to csv for FETCH
# --p directory path

import pandas as pd
import argparse
import glob

parser = argparse.ArgumentParser(description="""Heimdall candidates to csv for FETCH""")
parser.add_argument('-p','--dirpath',type=str,required=True)
parser.add_argument('-d','--dm',type=int,required=True)
parser.add_argument('-t','--inject_time', type=float,required=True)
args = parser.parse_args()

#combine all the candidates in a dataframe with the correct headings, save them in pre.csv
cnt=0
cand_lists=[]
for files in glob.glob(args.dirpath+"/"+"*cand"):
    filename = (glob.glob("*pulse.fil"))[0]
    try:
        cand_lists.append(pd.read_csv(files, header=None,\
                comment='#',delim_whitespace=True,\
                names=['snr','sample','tcand','width','dm_trial','dm',\
                'members','start_sample','end_sample']))
        cand_lists[cnt]["fil_file"]=args.dirpath+"/"+filename[:-4] + ".fil"
        cand_lists[cnt]["kill_mask"]=args.dirpath+"/"+filename[:-4] + "_badchans.txt"
        cand_lists[cnt]["label"]=0
        cand_lists[cnt]["num_files"]=1
        cnt+=1
    except:
        print("Zoinks! No cands my dude...")

#print(cnt) #note this count and the len(df) are not necessarily the same, cand files can be empty or contains multiple candidates
df=pd.concat(cand_lists)
df.to_csv("pre.csv")
total_cands=len(df)
print(f"Heimdall found {total_cands} candidates.")

#applying filter of all the candidates, save the results in phoenix.csv
#print("SUMMONING THE FILTER PHOENIX")
#we know the location of the injected pulse, so it may not differ by 0.1 seconds:
cond = (abs(df['tcand'] - (args.inject_time*0.000016)) < 0.1)
fdf=df[cond]
total_fdf=len(fdf)
fdf.to_csv('phoenix.csv',index=False, columns=['fil_file','snr','tcand','width','dm','label','num_files','kill_mask'], \
		header=["file","snr","stime","width","dm","label","num_files","chan_mask_path"])
print(f"The filter phoenix approves of {total_fdf}.")

