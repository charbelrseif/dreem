import csv
import math
import sys
import pandas as pd

numrows=0
energies=[]
capacity=0
rounding=-1
numpoints=0
numclass1=0

class1=''

#with open(sys.argv[1],'rb') as h5file:
    #csvreader = csv.reader(csvfile)
h5file = sys.argv[1]
df_binary = pd.read_hdf(h5file, h5file.replace('.h5', ''))
print(df_binary.columns)
#predictors = list(list(df_binary.columns[0:81])+list(df_binary.columns[85:180]))
#predictors.append(str(df_binary.columns[-1]))
#df_binary = df_binary[predictors]
#print(predictors)

for row in df_binary.iterrows():	
    numpoints=numpoints+1
    result = 0
    numrows=len(row[1][:-1])
    for elem in row[1][:-1]:
        result = result + float(elem) 
    c = row[1][-1]
    #print('c:', c)
    if (class1==''):
        class1=c
    if (c==class1):
        numclass1=numclass1+1
    if (rounding!=-1):
        result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
    energies=energies+[(result, c)]
    #print('energies:', energies)
    
print('\n')
sortedenergies=sorted(energies, key=lambda x: x[0])
print('sortedenergies:', sortedenergies)
curclass=sortedenergies[0][1]
print('curclass:', curclass)
changes=0
print('\n')
for item in sortedenergies:
	print('item: ', item)
	if (item[1]!=curclass):
		print('changes: ', changes)
		changes=changes+1
		curclass=item[1]
		print('changes: ', changes)

clusters=changes+1
mincuts=math.ceil(math.log(clusters)/math.log(2))
capacity=mincuts*numrows
#tmlpcap=mincuts*(numrows+1)+(mincuts+1)

# The following assume two classes!
print('changes:', changes)
entropy=-((float(changes)/numpoints)*math.log(float(changes)/numpoints)+(float(numpoints-changes)/numpoints)*math.log(float(numpoints-changes)/numpoints))/math.log(2)

print("Input dimensionality: ", numrows, ". Number of points:", numpoints, ". Class balance:", float(numclass1)/numpoints)
print("Eq. energy clusters: ", clusters, "=> binary decisions/sample:", entropy)
print("Max capacity need: ", (changes*(numrows+1))+changes,"bits")
print("Estimated capacity need: ",int(math.ceil(capacity)),"bits")

