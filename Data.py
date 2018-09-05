import pickle
d={};
def getData(fileName):
    count=0;
    X=[];
    Y=[];
    print("Hello");
    with open(fileName) as f:
        line=f.readline();
        while(line):
            line=f.readline();
            data=line.split(";");
            length = len(data);
            if(length==17):
                row=[];
                for i in range(length):
                    if(i==0 or i==5 or i==9 or i==11 or i==12 or i==13 or i==14):
                       row.append(int(data[i]));
                    else:
                        k=data[i].split("'");
                        if(k[1] in d.keys()):
                            row.append(d[k[1]]);
                        else:
                            d[k[1]]=count;
                            count=count+1;
                            row.append(d[k[1]]);
                #0 5 9 11 12 13 14
                X.append(row[0:length-1]);
                if(row[length-1]==d['no']):
                    Y.append([0.0,1.0]);
                else:
                    Y.append([1.0,0.0]);
    return X,Y;


if __name__=="__main__":
    X,Y=getData('bank-full.csv');
    pickle.dump(X,open('X','wb'));
    print(len(X));
    pickle.dump(Y,open('Y','wb'));