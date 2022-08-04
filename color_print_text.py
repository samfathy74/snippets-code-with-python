!pip install termcolor
from termcolor import colored, COLORS, cprint, HIGHLIGHTS
cprint(text='colorized text', color='red', on_color='gray')                #not need print
print(colored(text='colorized text', color='red', on_color='gray'))                #need print
