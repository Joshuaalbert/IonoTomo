
# coding: utf-8

# In[2]:

from RadioArray import RadioArray

def PrepareData():
    dataFile = 'DillPickleBug_data'
    
    print("creating radio array")
    radioArray = RadioArray()
    radioArray.log = None#solution to remove logger
    
    dataDict = {'radioArray':radioArray}
    
    f = open(dataFile,'wb')
    dill.dump(dataDict,f)
    f.close()
    return 

if __name__ == '__main__':
    PrepareData()
   
        


# In[ ]:



