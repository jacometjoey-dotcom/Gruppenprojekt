import streamlit as st


from Feature_01 import prime_Numbers 
from Feature_02 import even_numbers 




write = st.write



write("# Gruppenarbeit")
write("This is our Gruopprojekt, here should be all the work we do together.")
print(" I switched write.st to write for easier typing ")

write(prime_Numbers (20))

write(even_numbers(15))
