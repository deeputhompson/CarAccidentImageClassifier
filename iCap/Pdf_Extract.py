import PyPDF2
import collections
import tabula
from tabula import convert_into
import pandas as pd
import re
import os
import glob


counter=0
filepath=r"C:\Users\deepu\Desktop\iCAP\Work Directory\Subject"
os.chdir(filepath)
for file in glob.glob("*"):
    if(".PDF" in file):
        files=file.split(".")
        filename=files[0]
        print(filename)
        counter=counter+1
        if(filename=="Claim Details"):
            pdf1=os.path.abspath(file)
            df = tabula.read_pdf(pdf1,multiple_tables=True)
            Lpdf1=[]
            for i in df[0][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf1.append(k)
            del Lpdf1[0]
            col_list=[]
            for i in df[0][0]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                col_list.append(k)
            del col_list[0]
            data = {'0':Lpdf1}
            df = pd.DataFrame.from_dict(data,orient='index')
            df.columns = col_list
            df.transpose()
            print("Given Dataframe :\n", df)
            df.to_csv('ClaimsFraud_1.csv') 

            
        if(filename=="Claim Information"):
            pdf2=os.path.abspath(file)
            df = tabula.read_pdf(pdf2,multiple_tables=True)
            Lpdf2=[]
            for i in df[0][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf2.append(k)
            del Lpdf2[0]
            Lpdf22=[]
            for i in df[0][3]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf22.append(k)
            del Lpdf22[0]
            Lpdf2=Lpdf2+Lpdf22
            #print(Lpdf2)
            data = {'0':Lpdf2}
            df = pd.DataFrame.from_dict(data,orient='index')
            df.columns = Lpdf2
            df.transpose()
            print("Given Dataframe :\n", df)
            df.to_csv('ClaimsFraud_1.csv') 

        if(filename=="Claimant Data"):
            pdf3=os.path.abspath(file)
            df = tabula.read_pdf(pdf3,multiple_tables=True)
            Lpdf3=[]
            for i in df[0][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf3.append(k)
            del Lpdf3[0]
            #print(Lpdf3)
            Lpdf33=[]
            for i in df[0][3]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf33.append(k)
            del Lpdf33[0]
            #print(Lpdf3)

            Lpdf333=[]
            for i in df[1][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf333.append(k)
            del Lpdf333[0]
            #print(Lpdf333)

            Lpdf3333=[]
            for i in df[1][3]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub(" ", i)
                Lpdf3333.append(k)
            del Lpdf3333[0]
            Lpdf3=Lpdf3+Lpdf333+Lpdf33+Lpdf3333
            #print(Lpdf3)
            data = {'0':Lpdf3}
            df = pd.DataFrame.from_dict(data,orient='index')
            df.columns = Lpdf3
            df.transpose()
            print("Given Dataframe :\n", df)
            df.to_csv('ClaimsFraud_1.csv') 

        if(filename=="Claimant Details"):
            pdf4=os.path.abspath(file)
            df = tabula.read_pdf(pdf4,multiple_tables=True)
            #print(df)
            Lpdf4=[]
            for i in df[0][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub("", i)
                Lpdf4.append(k)
            del Lpdf4[0]
            #print(Lpdf4)
            Lpdf44=[]
            for i in df[1][1]:
                r_unwanted = re.compile("[\n\t\r]")
                k=r_unwanted.sub("", i)
                Lpdf44.append(k)
            del Lpdf44[0]
            #print(Lpdf44)
            Lpdf444=[]
            for i in df[2][1]:
                Lpdf444.append(i)
            del Lpdf444[0]
            del Lpdf444[9]
            #print(Lpdf444)
            Lpdf4444=[]
            for i in df[2][3]:
                Lpdf4444.append(i)
            del Lpdf4444[0]
            Lpdf4=Lpdf4+Lpdf44+Lpdf444+Lpdf4444
            #print(Lpdf4)
            data = {'0':Lpdf4}
            df = pd.DataFrame.from_dict(data,orient='index')
            df.columns = Lpdf4
            df.transpose()
            print("Given Dataframe :\n", df)
            df.to_csv('ClaimsFraud_1.csv')