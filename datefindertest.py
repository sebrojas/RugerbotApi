
import dateparser
matches = dateparser.parse('mañana a las 9pm y miercoles a las 2pm', languages=['es'])
print(matches)