import numpy as np
import cv2
L = 256
def FrequencyFiltering(imgin,H):
    #Ta không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)
    # Bước 1: DFT
    F = np.fft.fft2(f)
    # Bước 2: Dời F vào chính giữa ảnh
    F = np.fft.fftshift(F)
    # Bước 3: G = F*H
    G=F*H
    # Bước 4: Dời trở lại
    G = np.fft.ifftshift(G)
    # Bước 5: IDFT
    g = np.fft.ifft2(G)
    # Lấy phần thực
    gR = np.clip(g.real,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def Spectrum(imgin):
    #Ta không cần mở rộng ảnh có kích thươc s PxQ
    f = imgin.astype(np.float32)/(L-1)
    # Bước 1: DFT
    F = np.fft.fft2(f)
    # Bước 2: Dời F vào chính giữa ảnh
    F = np.fft.fftshift(F)
    # Bước 3: Tính phổ
    FR = F.real.copy()
    FI = F.imag.copy()
    S = np.sqrt(FR**2 + F.imag**2)
    S = np.clip(S,0,L-1)
    imgout = S.astype(np.uint8)
    return imgout

def CreateMoireFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0
    u1,v1 = 44,55
    u2,v2 = 85,55
    u3,v3 = 40,111
    u4,v4 = 81,111

    u5,v5 = M-44,N-55
    u6,v6 = M-85,N-55
    u7,v7 = M-40,N-111
    u8,v8 = M-81,M-111

    D0 = 10
    for u in range(0,M):
        for v in range(0,N):
            # u1,v1
            D = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u2,v2
            D = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u3,v3
            D = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u4,v4
            D = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u5,v5
            D = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u6,v6
            D = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u7,v7
            D = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if D<=D0:
                H.real[u,v]=0.0
            # u8,v8
            D = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if D<=D0:
                H.real[u,v]=0.0
    return H

def CreateInterferenceFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D0, M//2+D0+1):
                if v in range(N//2-D1,N//2+D1+1):
                    H.real[u,v] = 0.0
    return H

def CreateMotionFilter(M,N):
    H = np.zeros((M,N),np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi)<1.0e-6:
                phi = phi_prev
                RE = T
                IM = 0.0
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateDeMotionFilter(M,N):
    H = np.zeros((M,N),np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            mau_so = np.sin(phi)
            if abs(mau_so)<1.0e-6:
                phi = phi_prev
                RE = T
                IM = 0.0
            RE = phi/(T*np.sin(phi)/np.cos(phi))
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def RemoveMoire(imgin):
    M, N = imgin.shape
    H = CreateMoireFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def RemoveInterferenceFilter(imgin):
    M, N = imgin.shape
    H = CreateInterferenceFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout
    
def PlotMotionFilter(imgin):
    M,N = imgin.shape
    H = CreateMotionFilter(M,N)
    S = np.sqrt(H.real**2 + H.imag**2)
    S = S*(L-1)
    S = np.clip(S,0,L-1)
    imgout = S.astype(np.uint8)
    return imgout

def DeMotion(imgin):
    M, N = imgin.shape
    H = CreateDeMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

    # M,N=imgin.shape
    # # Bước 1 và 2
    # P = cv2.getOptimalDFTSize(M)
    # Q = cv2.getOptimalDFTSize(N)
    # fp = np.zeros((P,Q),np.float32)
    # fp[:M,:N] = 1.0*imgin/(L-1)
    # # Bước 3: Nhân fp với (-1)^(x+y)
    # for x in range(0,M):
    #     for y in range(0,N):
    #         if(x+y)%2 == 1:
    #             fp[x,y] = -fp[x,y]
    
    # # Bước 4: Tính DFT
    # F = cv2.dft(fp,flags=cv2.DFT_COMPLEX_OUTPUT)

    # # Tính spectrum
    # Spectrum = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    # Spectrum = np.clip(Spectrum, 0, L-1)
    # imgout = Spectrum.astype(np.uint8)
    # return imgout