import streamlit as st


from Feature_01 import prime_Numbers 



Test_List = [ i for i in range (1,100) ]



write = st.write



write("# Gruppenarbeit")
write("This is our Gruopprojekt, here should be all the work we do together.")
print(" I switched write.st to write for easier typing ")

write(prime_Numbers (Test_List))