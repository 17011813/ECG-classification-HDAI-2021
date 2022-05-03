# txt 파일을 csv 파일로 변환할 때 매 줄 마다 공백이 함께 저장 되는 문제 해결 --> line_terminator='\n' 를 추가해주면 된다.


import pandas as pd
forders = os.listdir("C://Users//yoona//Desktop//CMAPSSData")
print(forders)

df_all = pd.DataFrame()
for i in range(0,len(forders)):
    if forders[i].split('.')[1] == 'txt':
        file = 'C://Users//yoona//Desktop//CMAPSSData//' + forders[i]
        df= pd.read_csv(file, sep=" ")
        df.to_csv('C://Users//yoona//Desktop//공백없이 txt to csv로 변환//{}.csv'.format(forders[i].split('.')[0]), index=False, line_terminator='\n')
