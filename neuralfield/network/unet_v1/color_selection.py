import cv2
import numpy as np
# import video

def nothing(*arg):
        pass

def UI_select_colors():
    cv2.namedWindow( "result" ) # создаем главное окно
    cv2.namedWindow( "settings" ) # создаем окно настроек

    # cap = video.create_capture(0)
    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
    crange = [0,0,0, 0,0,0]

    while True:
        img = cv2.imread('.\\data\\img\\l12.png')        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')

        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        thresh = cv2.inRange(hsv, h_min, h_max)

        cv2.imshow('result', thresh) 
    
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    UI_select_colors()