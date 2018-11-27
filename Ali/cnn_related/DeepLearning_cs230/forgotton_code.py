


##############
# cnn server
#############
df = pd.read_excel("C:/Users/j70514/Documents/C4ISR/Staging Area/Table_Generated.xlsx", sheetname = 'T_ZM67PlanVsActual')
print("2")

objectCols = [df['Index'].tolist(), df['Level'].tolist(), df['Description'].tolist()]
data = zip(objectCols[0], objectCols[1], objectCols[2])

#for i,x in enumerate(data)
vals = [tuple_ for tuple_ in data ]

for i,val in enumerate(vals):
    vals[i]= {'1':str(val[0]),'2':str(val[1]),'3':str(val[2])}
#vals[:3]
finalresult = json.dumps(vals)

#ei.nodesArray={}
#tree = ei.generateTree(df, 0, df.iloc())
#print("3")

#arr =[]
##https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
#for key,value in ei.nodesArray.items():
#    arr.append({str(key):value.toString() #{"DFD": "FF" , "DFDS":3} 
#})

#finalresult = json.dumps(arr)