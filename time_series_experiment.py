import olmf_sequential as mf

url= './url_svmlight/'

def readUrlNormalized(file_number):
    filename=url+"Day"+str(file_number)+".svm"
    dataset=[]
    with open(filename) as f:
        for line in f:
            line_dict={}
            x=(line.rstrip()[3:]).split()
            y=int(line[:3])
            for elem in x:
                elem_list=elem.split(":")
                line_dict[int(elem_list[0])]=float(elem_list[1])
            line_dict['class_label']=int(y)
            dataset.append(line_dict)
    return dataset[:100]


dataset = readUrlNormalized(1)
olmf = mf.olmf(dataset)
olmf.fit()


