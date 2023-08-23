import commlib as cl
import matplotlib.pyplot as plt

TS = 1e-6
samples_per_symbol = 20
tinitial = 0
tguard = 10 * TS
name = 'Ilias Panagopoulos'

# ord function converts the character to it’s ASCII equivalent 
# format converts this to binary number 
# join is used to join each converted character to form a string. 
binary = ''.join(format(ord(i), '08b') for i in name)

for i in [4, 16]:
    # 4/16-QAM constellation
    constellation = cl.qam_constellation(i)

    # build the waveform
    wf = cl.digital_signal(TS = TS, samples_per_symbol = samples_per_symbol, tinitial = tinitial, tguard = tguard, constellation = constellation)
    
    # ensure the binary string is padded with zeros if its length is not already divisible
    if(len(binary) % (i ** 0.5) != 0):
        while(len(binary) % (i ** 0.5) != 0):
            binary += '0'
        
    # build waveform
    wf.modulate_from_bits(binary, constellation = constellation)
    wf.plot(False, i)

plt.show()