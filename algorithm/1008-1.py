# 링크: https://www.acmicpc.net/problem/1193
X=int(input())
i=0
mod=0
while True:
    if X<=0:
        break
    i+= 1
    mod = X
    X -= i
#print(i, mod)
if i % 2 == 1: # 홀수번째 카테고리면
    print("{}/{}".format(i+1-mod, mod)) #합은 i+1
else:
    print("{}/{}".format(mod, i+1-mod))