def fibo(n):
    temp = [0,1]
    if n >= 2:
        for i in range(2, n+1):
            temp.append(temp[i-2] + temp[i-1])
    return temp[n]
n=int(input())
print(fibo(n))