import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep 
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW) # Set pin 8 to be an output pin and set initial value to low (off)

while True: # Run forever
    GPIO.output(8, GPIO.HIGH) # Turn on
    sleep(0.5) # Sleep for 1 second
    GPIO.output(8, GPIO.LOW) # Turn off
    sleep(0.5) # Sleep for 1 second
