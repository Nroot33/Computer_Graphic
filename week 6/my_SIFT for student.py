import cv2
import numpy as np


def get_extrema(DoG, ext):
    for i in range(1, 4):
        for j in range(1, DoG.shape[0] - 1):
            for k in range(1, DoG.shape[1] - 1):
                # 최대값 혹은 최소값인 지점을 extrema로 구해주세요.
                target = DoG[j, k, i]
                max = target
                min = target

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if max < DoG[j + dj, k + dk, i + di]:
                                max = DoG[j + dj, k + dk, i + di]
                            if min > DoG[j + dj, k + dk, i + di]:
                                min = DoG[j + dj, k + dk, i + di]

                if target == max or target == min:
                    # xhat과 D(xhat)을 구하기 위한 미분을 수행해주세요.
                    dx = (DoG[j, k + 1, i] - DoG[j, k - 1, i]) / 2
                    dy = (DoG[j + 1, k, i] - DoG[j - 1, k, i]) / 2
                    ds = (DoG[j, k, i + 1] - DoG[j, k, i - 1]) / 2

                    dxx = (DoG[j, k + 1, i] - DoG[j, k - 1, i] - 2 * DoG[j, k, i])
                    dyy = (DoG[j + 1, k, i] - DoG[j - 1, k, i] - 2 * DoG[j, k, i])
                    dss = (DoG[j, k, i + 1] - DoG[j, k, i - 1] - 2 * DoG[j, k, i])

                    dxy = ((DoG[j + 1, k + 1, i] - DoG[j + 1, k - 1, i]) - (
                            DoG[j - 1, k + 1, i] - DoG[j - 1, k - 1, i])) / 4
                    dxs = ((DoG[j, k + 1, i + 1] - DoG[j, k - 1, i + 1]) - (
                            DoG[j, k + 1, i - 1] - DoG[j, k - 1, i - 1])) / 4
                    dys = ((DoG[j + 1, k, i + 1] - DoG[j, k, i - 1]) - (
                            DoG[j - 1, k, i + 1] - DoG[j - 1, k, i - 1])) / 4

                    H = np.asarray([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    dD = np.asarray([[dx], [dy], [ds]])
                    xhat = np.linalg.lstsq(-H, dD, rcond=-1)[0]
                    Dxhat = target + 0.5 * np.dot(dD.transpose(), xhat)

                    if np.absolute(xhat[0]) < 0.5 and np.absolute(xhat[1]) < 0.5 and np.absolute(xhat[2]) < 0.5 and np.absolute(Dxhat) > thresh:
                        trace = dxx + dyy
                        det = dxx * dyy - dxy ** 2

                        if det > 0 and (trace ** 2 / det) < ((r + 1) ** 2 / r) :
                            ext[j, k, i - 1] = 1

                # Thresholding을 수행해주세요. ( 적절한 위치만 ext 배열에 저장해주세요, )
    return ext


def SIFT(src, thresh, r):
    s = 1.3  # 초기 sigma
    a = 3.  # 극점을 찾을 이미지 수
    k = 2. ** (1 / a)  # scale step

    lv1sigma = np.array([s, s * k, s * (k ** 2), s * (k ** 3), s * (k ** 4), s * (k ** 5)])  # double image에 적용될 sigma.
    lv2sigma = np.array([s * (k ** 3), s * (k ** 4), s * (k ** 5), s * (k ** 6), s * (k ** 7), s * (k ** 8)])  # Original size image #start : 2 * sigma
    lv3sigma = np.array([s * (k ** 6), s * (k ** 7), s * (k ** 8), s * (k ** 9), s * (k ** 10), s * (k ** 11)])  # half size image #start : 4 * sigma
    lv4sigma = np.array([s * (k ** 9), s * (k ** 10), s * (k ** 11), s * (k ** 12), s * (k ** 13), s * (k ** 14)])  # quater size image #start : 8 * sigma

    # image resize
    doubled = cv2.resize(src, dsize=(0, 0), fx=2., fy=2., interpolation=cv2.INTER_LINEAR)  # 원본의 2배로 이미지를 resize 해주세요. cv2.INTER_LINEAR, cv2.INTER_NEAREST 자유롭게 사용.
    normal = src  # 원본과 동일
    half = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 가로 세로 각각 1/2
    quarter = cv2.resize(src, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)  # 가로 세로 각각 1/4

    # Gaussian 피라미드 저장할 3차원 배열
    lv1py = np.zeros((doubled.shape[0], doubled.shape[1], 6))
    lv2py = np.zeros((normal.shape[0], normal.shape[1], 6))
    lv3py = np.zeros((half.shape[0], half.shape[1], 6))
    lv4py = np.zeros((quarter.shape[0], quarter.shape[1], 6))

    print('make gaussian pyr')
    # Gaussian을 계산
    # ksize = 2 * int(4 * sigma + 0.5) + 1
    for i in range(6):
        # Gaussian Pyramids를 만들어주세요.
        # 예제에서는 한 Level(Octave)에 6개의 Gaussian Image가 저장됩니다.
        lv1img = cv2.GaussianBlur(doubled, (2 * int(4 * lv1sigma[i] + 0.5) + 1, 2 * int(4 * lv1sigma[i] + 0.5) + 1), lv1sigma[i])
        lv2img = cv2.GaussianBlur(doubled, (2 * int(4 * lv2sigma[i] + 0.5) + 1, 2 * int(4 * lv2sigma[i] + 0.5) + 1), lv2sigma[i])
        lv3img = cv2.GaussianBlur(doubled, (2 * int(4 * lv3sigma[i] + 0.5) + 1, 2 * int(4 * lv3sigma[i] + 0.5) + 1), lv3sigma[i])
        lv4lmg = cv2.GaussianBlur(doubled, (2 * int(4 * lv4sigma[i] + 0.5) + 1, 2 * int(4 * lv4sigma[i] + 0.5) + 1), lv4sigma[i])

        lv1py[:,:,i] = lv1img
        lv2py[:,:,i] = cv2.resize(lv2img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        lv3py[:,:,i] = cv2.resize(lv3img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        lv4py[:,:,i] = cv2.resize(lv4lmg, dsize=(0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR)

    # DoG 피라미드를 저장할 3차원 배열
    DoGlv1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    DoGlv2 = np.zeros((normal.shape[0], normal.shape[1], 5))
    DoGlv3 = np.zeros((half.shape[0], half.shape[1], 5))
    DoGlv4 = np.zeros((quarter.shape[0], quarter.shape[1], 5))

    print('calc DoG')

    # DoG를 계산
    for i in range(5):
        # Difference of Gaussian Image pyramids 를 구해주세요.
        DoGlv1[:,:,i] = lv1py[:,:,i+1] - lv1py[:,:,i]
        DoGlv2[:,:,i] = lv2py[:,:,i+1] - lv2py[:,:,i]
        DoGlv3[:,:,i] = lv3py[:,:,i+1] - lv3py[:,:,i]
        DoGlv4[:,:,i] = lv4py[:,:,i+1] - lv4py[:,:,i]

    # 극값의 위치를 표시할 3차원 배열
    extPy1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    extPy2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    extPy3 = np.zeros((half.shape[0], half.shape[1], 3))
    extPy4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    # Extrema의 위치 계산
    print('find extrema')
    extPy1 = get_extrema(DoGlv1, extPy1)
    extPy2 = get_extrema(DoGlv2, extPy2)
    extPy3 = get_extrema(DoGlv3, extPy3)
    extPy4 = get_extrema(DoGlv4, extPy4)

    extr_sum = extPy1.sum() + extPy2.sum() + extPy3.sum() + extPy4.sum()
    extr_sum = extr_sum.astype(np.int)
    keypoints = np.zeros((extr_sum, 3))  # 원래는 3가지의 정보가 들어가나 과제에선 Y좌표, X 좌표, scale 세 가지의 값만 저장한다.

    # 값 저장
    count = 0  # keypoints 수를 Count

    for i in range(3):
        for j in range(doubled.shape[0]):
            for k in range(doubled.shape[1]):
                # Lv1
                # Keypoints 배열에 Keypoint의 정보를 저장하세요. 함수로 만들어서 수행하셔도 됩니다.
                if extPy1[j, k, i] == 1:
                    keypoints[count, :] = np.array([j/2., k/2., lv1sigma[i*2]])
                    count += 1

    for i in range(3):
        for j in range(normal.shape[0]):
            for k in range(normal.shape[1]):
                # Lv2
                # Keypoints 배열에 Keypoint의 정보를 저장하세요.
                if extPy2[j, k, i] == 1:
                    keypoints[count, :] = np.array([j, k, lv2sigma[i*2]])
                    count += 1

    for i in range(3):
        for j in range(half.shape[0]):
            for k in range(half.shape[1]):
                # Lv3
                # Keypoints 배열에 Keypoint의 정보를 저장하세요.
                if extPy3[j, k, i] == 1:
                    keypoints[count, :] = np.array([j*2., k*2., lv3sigma[i*2]])
                    count += 1

    for i in range(3):
        for j in range(quarter.shape[0]):
            for k in range(quarter.shape[1]):
                # Lv4
                # Keypoints 배열에 Keypoint의 정보를 저장하세요.
                if extPy4[j, k, i] == 1:
                    keypoints[count, :] = np.array([j*4., k*4., lv4sigma[i*2]])
                    count += 1

    return keypoints


if __name__ == '__main__':
    src = cv2.imread('.\\building.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.double)
    gray /= 255.

    thresh = 0.03
    r = 10.  # 원 논문에서 값을 10으로 사용

    keypoints = SIFT(gray, thresh=thresh, r=r)

    for i in range(len(keypoints)):
        cv2.circle(src, (int(keypoints[i, 1]), int(keypoints[i, 0])), int(1 * keypoints[i, 2]), (0, 0, 255),
                   1)  # 해당 위치에 원을 그려주는 함수

    src2 = cv2.imread('.\\building_temp.jpg')
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    gray2 = gray2.astype(np.double) / 255.

    keypoints2 = SIFT(gray2, thresh=thresh, r=r)

    for i in range(len(keypoints2)):
        cv2.circle(src2, (int(keypoints2[i, 1]), int(keypoints2[i, 0])), int(1 * keypoints2[i, 2]), (0, 0, 255),
                   1)  # 해당 위치에 원을 그려주는 함수

    cv2.imshow('src', src)
    cv2.imshow('src2', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()
