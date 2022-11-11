import rfifind
import os
import glob
import argparse

# identify_badchans_py3
# identifies zapped channels for heimdall from rfifind and CREATES summonheimdall.sh
# --l low DM value
# --d high DM value
# --f filename

x=glob.glob("*rfifind.mask")[0]
mask = rfifind.rfifind(x)
base=x.split("_")[1]+"_"+x.split("_")[2]

parser = argparse.ArgumentParser(description="""Identify zapped channels for heimdall from rfifind and write heimdall summoing command""")
parser.add_argument('-l','--lodm',type=int,required=True)
parser.add_argument('-d','--hidm',type=int,required=True)
parser.add_argument('-f','--filename',type=str,required=True)
args = parser.parse_args()

badbase=args.filename.split(".")[0]+"_badchans.txt"
chans = ""
commachans = ""
for chan in mask.mask_zap_chans:
    flip=False
    if flip:
        # - 1 because of zero-based-indexing
        chans += str(mask.nchan - chan - 1) + " "
        commachans += str(mask.nchan - chan -1) + ","
    else:
        chans += str(chan) + " "
        commachans += str(chan) + " ,"

badbois=[]
flip_ls=[]

for i in chans[:-1].split(" "):
    i=int(i)
    flip_ls.append(127-i)
flip_ls.sort()

cmd=""
i=0
print(flip_ls)
while i < len(flip_ls):

    first=flip_ls[i]
    if first == flip_ls[-1]:
        badbois.append(first)
        cmd+="-zap_chans "+str(first)+" "+str(first)+" "
        break
    
    second=flip_ls[i+1]
    
    if first+1!=second:
        cmd+="-zap_chans "+str(first)+" "+str(first)+" "
        badbois.append(first)
        i+=1

    else:
        j=i
        while flip_ls[j] + 1 == flip_ls[j+1]:
            if  flip_ls[j+1] == flip_ls[-1]:
                break
            badbois.append(flip_ls[j])
            j+=1
            
        cmd+="-zap_chans "+str(first)+" "+str(flip_ls[j])+" "
        badbois.append(flip_ls[j])
        i=j+1

out_txt_file = "./"+badbase
txtfile = open(out_txt_file, "w")
for i in badbois:
    txtfile.write(str(i)+"\n")
txtfile.close()

cmd = "heimdall -nsamps_gulp 262144 -dm "+str(args.lodm)+" "+str(args.hidm)+" -boxcar_max 1024 -cand_sep_dm_trial 200 -cand_sep_time 128 -cand_sep_filter 3 -dm_tol 1.05 -detect_thresh 7 -dm_pulse_width 16 " +cmd+"-f "+args.filename#+base+"_8bit.fil"

print(cmd)
print(str(100*len(badbois)/128)+'%')

txtfile = open("./rfi_sum.txt", "w")
txtfile.write(str(len(badbois))+" ")
txtfile.close()

txtfile = open("./summonheimdall.sh", "w")
txtfile.write(cmd)
txtfile.close()
