import numpy as np
import cv2

L=256

def Negative(imgin):
    # M là độ cao của ảnh, N là độ rộng của ảnh
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    # Quét ảnh
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            s=L-1-r
            imgout[x,y]=np.uint8(s)
    return imgout

def Negative_Color(imgin):
    # M là độ cao của ảnh, N là độ rộng của ảnh, C: channel: kênh và C=3 cho ảnh (R,G,B)
    M,N,C=imgin.shape
    imgout=np.zeros((M,N,C),np.uint8)
    # Quét ảnh
    for x in range(0,M):
        for y in range(0,N):
            # Ảnh opencv là BGR
            b=imgin[x,y,0]
            b=L-1-b

            g=imgin[x,y,1]
            g=L-1-g

            r=imgin[x,y,2]
            r=L-1-r

            imgout[x,y,0]=np.uint8(b)
            imgout[x,y,1]=np.uint8(g)
            imgout[x,y,2]=np.uint8(r)
    return imgout

def Logarit(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    c=(L-1)/np.log(1.0*L)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            if r == 0:
                r=1
            s=c*np.log(1.0+r)
            imgout[x,y]=np.uint8(s)
    return imgout

def Power(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    gamma=5.0
    c=np.power(L-1,1.0-gamma)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            if r == 0:
                r=1
            s=c*np.power(r,gamma)
            imgout[x,y]=np.uint8(s)
    return imgout

def PiecewiseLinear(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    rmin,rmax,_,_=cv2.minMaxLoc(imgin)
    r1=rmin
    s1=0
    r2=rmax
    s2=L-1
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            # Đoạn I
            if r<r1:
                s=1.0*(s1/r1)*r
            # Đoạn II
            elif r<r2:
                s=1.0*((s2-s1)/(r2-r1))*(r-r1)+s1
            # Đoạn III
            else:
                s=1.0*(L-1-s2)/(L-1-r2)*(r-r2)+s2  
            imgout[x,y]=np.uint8(s)
    return imgout

def Histogram(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,L),np.uint8)+255
    h=np.zeros(L,np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            h[r]=h[r]+1
    p=h/(M*N)
    scale=3000
    for r in range(0,L):
        cv2.line(imgout,(r,M-1),(r,M-1-np.int32(scale*p[r])),(0,0,0))
    return imgout

def HistEqual(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)

    h=np.zeros(L,np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            h[r]=h[r]+1
    p=1.0*h/(M*N)

    s=np.zeros(L,np.float32)
    
    for k in range(0,L):
        for j in range(0,k+1):
            s[k]=s[k]+p[j]
    
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            imgout[x,y]=np.uint8((L-1)*s[r])
    return imgout

def HistEqualColor(imgin):
    img_b = imgin[:,:,0]
    img_g = imgin[:,:,1]
    img_r = imgin[:,:,2]
    img_b = cv2.equalizeHist(img_b)
    img_g = cv2.equalizeHist(img_g)
    img_r = cv2.equalizeHist(img_r)

    imgout = imgin.copy()
    imgout[:,:,0]=img_b
    imgout[:,:,1]=img_g
    imgout[:,:,2]=img_r
    return imgout

def LocalHist(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    m=3
    n=3
    a=m//2
    b=n//2
    for x in range(a,M-a):
        for y in range(b,N-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x,y]=w[a,b]
    return imgout

def HistStat(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    m=3
    n=3
    a=m//2
    b=n//2
    mean,stdtev=cv2.meanStdDev(imgin)
    mG=mean[0,0]
    sigmaG=stdtev[0,0]
    C = 22.8
    k0=0.0
    k1=0.1
    k2=0.0
    k3=0.1
    for x in range(a,M-a):
        for y in range(b,N-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            mean,stdtev=cv2.meanStdDev(w)
            msxy=mean[0,0]
            sigmasxy=stdtev[0,0]
            if (k0*mG<=msxy<=k1*mG) and (k2*sigmaG<=sigmasxy<=k3*sigmaG):
                imgout[x,y]=np.uint8(C*imgin[x,y])
            else:
                imgout[x,y]=imgin[x,y]
    return imgout

def SmoothGauss(imgin):
    m = 43
    n = 43
    sigma = 7.0
    w=np.zeros((m,n),np.float32)
    a = m // 2
    b = n // 2
    for s in range(-a,a+1):
        for t in range(-b,b+1):
            r=np.exp(-(s*s+t*t)/(2*sigma*sigma))
            w[s+a,t+b] = r
    K=np.sum(w)
    w = w/K
    imgout=cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout

def CreateImpulseNoise(imgin):
    M, N = imgin.shape
    np.random.seed(100)
    n = 1000
    imgout = imgin.copy()
    for i in range(0,n):
        x = np.random.randint(10,M-10)
        y = np.random.randint(10,N-10)
        r = np.random.randint(0, 5)
        imgout[x,y] = np.uint8(r)
        imgout[x-1,y-1] = np.uint8(r)
        imgout[x+1,y+1] = np.uint8(r)

        x = np.random.randint(10,M-10)
        y = np.random.randint(10,N-10)
        r = np.random.randint(250, 255)
        imgout[x,y] = np.uint8(r)
        imgout[x-1,y-1] = np.uint8(r)
        imgout[x+1,y+1] = np.uint8(r)

    return imgout

def Sharp(imgin):
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    Laplacian = cv2.filter2D(imgin,cv2.CV_32FC1,w)
    imgout = imgin - Laplacian
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout