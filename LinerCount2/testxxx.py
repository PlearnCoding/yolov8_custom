import RPi.GPIO as GPIO
import time

relay_pin1 = 24
relay_pin2 = 16
sensor_pin = 25

GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin1, GPIO.OUT)
GPIO.setup(relay_pin2, GPIO.OUT)
GPIO.setup(sensor_pin, GPIO.IN)

#GPIO.output(relay_pin2, False)
GPIO.output(relay_pin1, False)


count = 0
while True:
    print('count = ',count)
    toc = time.time()
    while GPIO.input(sensor_pin)==1 : #have object
        #GPIO.output(relay_pin2, False)
        GPIO.output(relay_pin1, True) # relay off ,stopper off
        time.sleep(0.1)
        print('sensor = 1','relay = Ture')
        if time.time()-toc > 3:
            print('every 3 sec -> sensor = 1','relay = Ture')
            toc = time.time()
    count +=1
    while GPIO.input(sensor_pin)==0 : # sensor on ,no object
        #GPIO.output(relay_pin2, True)
        GPIO.output(relay_pin1, False) # relay on ,stopper on
        time.sleep(0.5)
        print('sensor = 0','relay = False')
'''
for i in range(10):
    GPIO.output(relay_pin, True)
    #if GPIO.input(sensor_pin):
    print(GPIO.input(sensor_pin))
    time.sleep(2)
    GPIO.output(relay_pin, False)
    time.sleep(2)
print('finish')
'''