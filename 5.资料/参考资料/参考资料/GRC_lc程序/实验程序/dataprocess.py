def intTo2Str(X,K):
    try:
        x=long(X)
    except:
        x=0
    try:
        K=int(X)
    except:
        K=0
    if K<1:
        K=1
    if X<0:
        FH=1;X=-X
    else:
        FH=0
    A=[0 for J in xrange(0,K)]
    J=K-1
    while(J>=0)and (X>0):
        Y=X%2
        X=X/2
        A[J]=Y
        J=J-1
    if FH==1:
        for J in xrange(0,K):
            if A[J]==1:
                A[J]=0
            else:
                A[J]=1
        J=K-1
        while J>=0:
            A[J]=A[J]+1
            if A[J]<=1:
                break;
            A[J]=0
            J=J-1
    return "".join([chr(J+48) for J in A])
############################################
f=open(r"/home/leichao/桌面/code.txt",'wb')
f1=open(r"/home/leichao/桌面/send.txt")
send=f1.readlines()
print send
send=send[0][0:-1]
##send=bin(send).replace('0b','')
print str(send)
f1.close
length=len(send)
##
serial =[1,0,1,1,0,1,0,0]
serial=serial*2
for i in serial:
    f.write(chr(i))
##
str1 =intTo2Str(length,8)
for i in range(8):
    temp = str1[i]
    temp = int(temp)
    f.write(chr(temp))
##
for i in range(length):
    ##str1 = intTo2Str(ord(send[i]),8)
    str1=bin(ord(send[i])).replace('0b','')
    m=len(str1)
    for i in range(m):
        temp = str1[i]
        temp = int(temp)
        f.write(chr(temp))
##
serial =[1,1,1,1,1,1,1,1]
for i in serial:
    f.write(chr(i))
f.close()
