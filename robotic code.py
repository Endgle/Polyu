import cv2
import numpy as np
import Adafruit_PCA9685  
import time
import RPi.GPIO as GPIO

# Initialize PCA9685 servo controller
pwm = Adafruit_PCA9685.PCA9685(0x41)  
pwm.set_pwm_freq(50) 

# Initialize GPIO for motor control
IN1 = 24
IN2 = 23
ENA = 18

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# Motor control functions
def motor_on():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("FIRE!")
    
def motor_off():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(ENA, GPIO.LOW)
    print("Motor OFF")

# Servo control function
def set_servo_angle(channel, angle):
    pulse = 4096 * ((angle * 11) + 500) / 20000  
    pwm.set_pwm(channel, 0, int(pulse))

# Initialize servo positions
set_servo_angle(1, 90)
set_servo_angle(2, 90)
set_servo_angle(3, 85)

# Initialize camera
cap = cv2.VideoCapture(0)

# Set camera parameters
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.01)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
cap.set(cv2.CAP_PROP_CONTRAST, 70)
cap.set(cv2.CAP_PROP_SATURATION, 70)

# Green color range definition
green_lower = np.array([50, 50, 120])
green_upper = np.array([85, 255, 255])

cap.set(3, 640) 
cap.set(4, 480) 

print("press s to save")
print("press q to exit")

# Initialize variables
currentAngle_x = 90  
currentAngle_y = 90
last_fire_time = 0
fire_cooldown = 3  # Fire cooldown time in seconds
current_distance = 0  # Initialize distance variable

# Offset to compensate for trajectory
horizontal_offset = 26  # Negative value moves aim point left
vertical_offset = -130   # Positive value moves aim point down

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        original = frame.copy()
      
        # Apply gamma correction to reduce brightness
        frame_float = frame.astype(np.float32) / 255.0
        gamma = 1.5
        frame_adjusted = np.power(frame_float, gamma)
        frame = (frame_adjusted * 255).astype(np.uint8)
        
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create green mask
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        cv2.imshow('Green Mask', mask)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        cv2.imshow('Processed Mask', mask)
        
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        h, w = frame.shape[:2]
        
        # Draw crosshairs with offset applied
        center_x = w // 2 + horizontal_offset
        center_y = h // 2 + vertical_offset
        
        cv2.line(res, (center_x, 0), (center_x, h), (255, 0, 0), 1)  
        cv2.line(res, (0, center_y), (w, center_y), (255, 0, 0), 1) 

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_detected = False
        
        for c in contours:
            area = cv2.contourArea(c)
            
            if area > 300 and area < 5000:  
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                
                if len(approx) == 4:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    width = rect[1][0]
                    height = rect[1][1]
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                    
                    if 0.7 < aspect_ratio < 1.3:
                        contour_mask = np.zeros_like(mask)
                        cv2.drawContours(contour_mask, [c], -1, 255, -1)

                        # Calculate average HSV values
                        mean_val = cv2.mean(hsv, mask=contour_mask)
                        hue, saturation, value = mean_val[0], mean_val[1], mean_val[2]
                        
                        # Check if HSV values are within green range
                        if green_lower[0] <= hue <= green_upper[0] and \
                           green_lower[1] <= saturation <= green_upper[1] and \
                           green_lower[2] <= value <= green_upper[2]:
                            
                            # Draw green box
                            cv2.drawContours(res, [box], 0, (0, 255, 0), 3)
                            cv2.drawContours(original, [box], 0, (0, 255, 0), 3)
                            
                            M = cv2.moments(c)
                            if M["m00"] != 0:  
                                cx = int(M["m10"] / M["m00"])  
                                cy = int(M["m01"] / M["m00"])  
                                cv2.circle(res, (cx, cy), 5, (255, 0, 0), -1)
                                cv2.circle(original, (cx, cy), 5, (255, 0, 0), -1)
                
                                object_detected = True
                                
                                target_center = [cx, cy]
                                
                                # Calculate offset from adjusted center (with offset applied)
                                offset_x = cx - center_x  
                                offset_y = cy - center_y 
                                
                                # Calculate distance from adjusted center
                                current_distance = np.sqrt(offset_x**2 + offset_y**2)
                                
                                # Dynamic servo speed adjustment
                                # Faster when far, slower when close
                                if current_distance > 150:  # Far distance
                                    k_p = 0.01  # Faster speed
                                elif current_distance > 40:  # Far distance
                                    k_p = 0.01
                                elif current_distance > 15:  # Medium distance
                                    k_p = 0.01  # Medium speed
                                else:  # Close distance
                                    k_p = 0.01  # Slower speed
                                
                                # Check if should fire
                                if current_distance < 6:  # When close enough to target
                                    current_time = time.time()
                                    print("Target <4")
                                    if current_time - last_fire_time > fire_cooldown:
                                        time.sleep(0.05)
                                        print("has cooldown.")
                                        #if distance > 8:
                                        #    break
                                        print("Target locked! Firing...")
                                        motor_on()
                                        time.sleep(0.1)  # Fire duration
                                        motor_off()
                                        motor_on()
                                        time.sleep(0.1)  # Fire duration
                                        motor_off()
                                        motor_on()
                                        time.sleep(0.1)  # Fire duration
                                        motor_off()
                                        last_fire_time = current_time
                                else:
                                    # Move servos
                                    new_angle_x = currentAngle_x - offset_x * k_p
                                    new_angle_y = currentAngle_y - offset_y * k_p
                                    
                                    new_angle_x = max(0, min(180, new_angle_x))
                                    new_angle_y = max(0, min(180, new_angle_y))
                                    
                                    set_servo_angle(1, new_angle_x)
                                    set_servo_angle(2, new_angle_y)
                                    currentAngle_x = new_angle_x
                                    currentAngle_y = new_angle_y
                                
                                break
        
        # Display distance information on all frames - USING OLD STRING FORMATTING
        distance_text = "Distance: {:.2f}".format(current_distance)
        status_text = "Target Locked" if object_detected else "Searching"
        
        # Put text on original frame
        cv2.putText(original, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(original, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if object_detected else (0, 0, 255), 2)
        cv2.putText(original, "Servo X: {:.1f}".format(currentAngle_x), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(original, "Servo Y: {:.1f}".format(currentAngle_y), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Put text on result frame
        cv2.putText(res, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(res, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if object_detected else (0, 0, 255), 2)
        
        # Put text on adjusted frame
        cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if object_detected else (0, 0, 255), 2)
        
        cv2.imshow('Original', original)  
        cv2.imshow('Result', res)         
        cv2.imshow('Adjusted Frame', frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):  
            cv2.imwrite('captured_image.png', original)
            print("saved!")
        elif k == ord('d'):  
            break

except KeyboardInterrupt:
    print("Program interrupted")

finally:
    # Clean up resources
    cap.release()          
    cv2.destroyAllWindows() 
    set_servo_angle(1, 90)
    set_servo_angle(2, 90)
    set_servo_angle(3, 90)
    motor_off()
    GPIO.cleanup()