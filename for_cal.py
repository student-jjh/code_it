
print(0.2659279778393353/2)
def gini(a,b):

    sum_num=a+b

    return (1-(a/sum_num)**2-(b/sum_num)**2)


for_answer=[]

k=int(input(""))

for i in range(k):
    a,b=map(int,(input("").split()))
    for_answer.append(gini(a,b))

print((sum(for_answer)/len(for_answer)))