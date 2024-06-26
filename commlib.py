import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.special import erfc

DEFAULT_PLOT_SETTINGS = {
    'plot_type' : '-', 
    'xlabelt' : 't', 
    'ylabelt' : 'x(t)',
    'xlabelf' : 'f', 
    'ylabelf' : 'X(f)',    
    'xlimt' : None,
    'ylimt' : None,
    'xlimf' : None,
    'ylimf' : None,    
    'show_gridt' : False,
    'show_gridf' : False,    
    'titlet' : None,
    'titlef' : None    
}

def array_to_str(a):
    astr = [ str(x) for x in a ]
    return ''.join(astr)

def str_to_array(s):
    return np.array( [ int(x) for x in s ] ).astype(int)

# function to create the square pulse
def square(t , T):    
    f = np.logical_and( t < T / 2.0 , t >= -T/2.0 )
    return f.astype('float')

# Calculate a simple cosine sinal
def cos_signal(A, f0, t, phi = 0.0 ):
    return A * np.cos ( 2.0 * np.pi * f0 * t + phi )

# create time axis
def time_axis(Tmin, Tmax, N):   
    return np.linspace(Tmin, Tmax, N, endpoint = False)

# Plot signal
def plot_signal(t, x, plot_type = 'o', close_all = False,
                      xlabel = 't', ylabel = 'x(t)', figure_no = None,
                      xlim = None, ylim = None, show_grid = False, title = None):
    
    if close_all:
        plt.close('all')
        
    if figure_no is None:
        plt.figure()
    else:
        plt.figure(figure_no)
    
    plt.plot( t, x, plot_type )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel ) 
    
    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)
    
    if show_grid:
        plt.grid()
        
    if title is not None:
        plt.title(title)

# random_bits : generation of random bits with equal probability
def random_bits(Nbits):
    return np.random.randint(0, high = 2, size = Nbits, dtype = int)

# gray coding        
def gray_code(m):

    if m==1:
        g = ['0','1']

    elif m>1:
        gs = gray_code(m-1)
        gsr = gs[::-1]
        gs0 = ['0' + x for x in gs]
        gs1 = ['1' + x for x in gsr]
        g= gs0 + gs1
    return g         

def str_to_bitsarray( bits_str ):    
    bits = np.zeros( len(bits_str) )
    for i, bit in enumerate(bits_str):
        bits[i] = int( bit )
    
    return bits

def Qfunction(x):
    return 0.5 * erfc( x / np.sqrt(2) )

class signal:
    
    def __init__(self, t = None, samples = None, signal_callable = None):
        self.t = t               
        if self.t is not None:
            self.Dt = t[1] - t[0]
            self.N = t.size
            
        if samples is not None:
           self.samples = samples
        elif signal_callable is not None:
           self.signal = signal_callable(t)
          
        self.set_default_plot_properties()
        
    def set_time_axis(self, Tmin, Tmax, N):
        self.t = np.linspace(Tmin, Tmax, N, endpoint = False)
        self.N = N
        self.Dt = self.t[1] - self.t[0]
        return self.t
    
    def set_frequency_axis(self):
        n = np.arange(-self.N / 2.0, self.N / 2.0, 1)
        self.Df = 1.0 / ( self.N * self.Dt)
        self.f = n * self.Df
        return self.f
    
    def calc_spectrum(self):
        self.set_frequency_axis()
        self.spec = self.Dt * np.fft.fftshift( 
                              np.fft.fft( np.fft.fftshift( self.samples ) ) )
        return self.spec
        
    def calc_invspectrum(self):
        self.samples = self.N * self.Df * np.fft.fftshift(
                              np.fft.ifft( np.fft.fftshift( self.spec ) ) )
        return self.samples
    
    def energy(self):
        return np.trapz( np.abs( self.samples ) ** 2.0, self.t)
                
    def average_power(self):
        ta = np.min(self.t)
        tb = np.max(self.t)
        return self.energy() / ( tb - ta )

    def power_density(self):
        T = np.max(self.t) - np.min(self.t)
        
        if hasattr( self, 'spec' ):
            self.calc_spectrum()
            
        return 1.0 / T * np.abs( self.spec ) ** 2.0

    def set_default_plot_properties( self ):
        
        for key in DEFAULT_PLOT_SETTINGS:
            setattr( self, key, DEFAULT_PLOT_SETTINGS[key] )
        
    def plot(self, close_all = False, figure_no = None, what = 'time'):
        
        if what == 'time':
           plot_signal(self.t, self.samples, plot_type = self.plot_type, 
                       close_all = close_all, xlabel = self.xlabelt, 
                       ylabel = self.ylabelt, figure_no = figure_no,
                       xlim = self.xlimt, ylim = self.ylimt, 
                       show_grid = self.show_gridt, title = self.titlet)
        
        elif what == 'spec':
           plot_signal(self.f, self.spec, plot_type = self.plot_type, 
                       close_all = close_all, xlabel = self.xlabelf, 
                       ylabel = self.ylabelf, figure_no = figure_no,
                       xlim = self.xlimf, ylim = self.ylimf, 
                       show_grid = self.show_gridt, title = self.titlef)
    
    def windowed(self, ta, tb):
                
        j = np.where( np.logical_and(self.t >= ta, self.t <= tb ) )
        y = np.zeros( self.N )
        y[j] = self.samples[j]
        
        return signal(t = self.t, samples = y)    
    
    def frequency_axis(self):
        return self.frequency_axis
      
    def __add__(self, sig):
        if not np.array_equal(self.t , sig.t):
            raise ValueError('time axis must be the same in order for signals to be added')
            
        return signal(t = self.t, samples = self.samples + sig.samples)
    
    def get_spectrum(self):
        return self.f, self.spec
    
class square_pulse(signal):
    
    def __init__(self, t, T1, tcenter = 0.0):
        samples = square(t - tcenter, T1)
        super().__init__( t = t, samples = samples )        

class carrier(signal):
    
    def __init__(self, t, f0, A = 1, phi = 0.0):
        samples = cos_signal(A, f0, t, phi = phi )
        super().__init__( t = t, samples = samples )

class constellation:
    
    def __init__(self, title = None):
       self.bit_map = {}
       self.bits = []
       self.bits_str = []
       self.symbols = []
       self.title = title
              
    def avg_power(self):
        return np.mean( np.abs(self.symbols) ** 2.0)
    
    def plot(self, figure_no = None, plot_type = 'o'):
    
        if figure_no is None:
            plt.figure()
        else:
            plt.figure(figure_no)
        
        cr = np.real(self.symbols)
        ci = np.imag(self.symbols)
        plt.plot(cr, ci, plot_type)
        plt.xlabel('Real')
        plt.ylabel('Imag')
        
        if self.title is not None:
            plt.title(self.title)
    
    def plot_map( self, figure_no = None, disp_x = 0.0, disp_y = 0.0,
                  rotation = 90, plot_type = 'bo', axis_equal = True):
    
        self.plot( figure_no = figure_no, plot_type = plot_type)
        
        for i, bits in enumerate(self.bits_str):
            symbol = self.symbols[i]
            bits_str = self.bits_str[i]
            plt.text( np.real(symbol) + disp_x, np.imag(symbol) + disp_y, bits_str, 
                      rotation = rotation)
        
        if self.title is not None:
            plt.title(self.title)
            
        if axis_equal:
            plt.axis('equal')
            
    def set_symbols( self, symbols ):
        self.symbols = symbols
        
    def set_gray_bits( self, m ):
        g = gray_code( m )
        self.bits = []
        self.bits_str = []
        self.map = {}
        self.m = m
        
        for i, cw in enumerate(g):
            self.bits_str.append(cw)
            self.bits.append( str_to_bitsarray( cw ) )
            self.map[ cw ] = self.symbols[ i ]
            
    def bits_to_symbols( self, bits, return_groups = False ):
        symbols = []
        bitgroups = []
        i = 0
        j = 0

        if not isinstance(bits, str):
            bits = array_to_str(bits)
                        
        while i < len(bits):

            key = bits[ i : i + self.m ]
            bitgroups.append( key )
            symbols.append( self.map[ key ] )
            i += self.m
            j += 1

        if not return_groups:    
            return np.array(symbols)    
        else:
            return np.array(symbols), bitgroups
    
    def find_closest( self, sample ):
        return np.abs( self.symbols - sample ).argmin()            
    
    def decode( self, sample ):
        i = self.find_closest( sample )
        return [self.symbols[i], self.bits[i], self.bits_str[i] ]
    
class pam_constellation(constellation):
    
    def __init__(self, M, beta = 1, title = None, SNRbdB = None):
        super().__init__(title = title)
        
        self.M = M
        self.m = np.log2(M).astype(int)
        self.SNRbdB = SNRbdB
        
        symbols = np.zeros( M )
        for i in range( M ):
            symbols [ i ] = 2 * i - M + 1
            
        self.set_symbols( symbols )
        self.set_gray_bits( self.m )
        
    def ser(self):
        SNRb = 10 ** ( self.SNRbdB / 10)
        q = 6 * SNRb * self.m / (self.M ** 2.0 - 1)
        return 2 * (self.M-1) / self.M * Qfunction ( np.sqrt(q) )
    
    def ber(self):
        return self.ser() / self.m

#---qamconstellation1
class qam_constellation(constellation):

    def set_gray_bits( self, m ):
        ms = int(m / 2)
        g = gray_code( ms )
        self.bits = []
        self.bits_str = []
        self.map = {}
        self.m = m
        i = 0
        
        for p, cwp in enumerate(g):
            for q, cwq in enumerate(g):
                cw = cwp + cwq                
                self.bits_str.append(cw)
                self.bits.append( str_to_bitsarray( cw ) )
                self.map[ cw ] = self.symbols[ i ]
                i += 1
                
    def __init__(self, M, beta = 1, title = None, SNRbdB = None):
        super().__init__(title = title)
        
        self.M = M
        self.m = np.log2(M).astype(int)
        self.SNRbdB = SNRbdB
        Ms = np.sqrt(M).astype(int)
        self.Ms = Ms
        i = 0
        symbols = np.zeros( M, dtype = complex )
        for p in range( Ms ):
            for q in range( Ms ):
                symbols [ i ] = beta * ( 2*p - Ms + 1 ) + 1j * beta * ( 2*q - Ms + 1 )
                i += 1
                
        self.set_symbols( symbols )        
        self.set_gray_bits( self.m )
        
    def ser(self):
        SNRb = 10 ** ( self.SNRbdB / 10)
        q2 = 3 * SNRb * self.m / (self.M - 1)
        q = np.sqrt(q2)
        P = 2 * ( np.sqrt(self.M) - 1 ) / np.sqrt(self.M) * Qfunction(q)
        return 1 - (1-P) ** 2

    def ber(self):
        return self.ser() / np.log2(self.M) 
#---qamconstellation2
    
#---pskconstellation1                           
class psk_constellation(constellation):

    def __init__(self, M, R = 1,title = None, SNRbdB = 10):
        super().__init__(title = title)
        
        self.M = M
        self.m = np.log2(M).astype(int)
        self.SNRbdB = SNRbdB
        self.R = R
        self.SNRb = 10 ** (SNRbdB/10)
        self.SNRS = self.SNRb * self.m
        self.vmax = 200
        self.Nphi = 1000
        
        symbols = np.zeros( M, dtype = complex )
        for i in range( M ):
            symbols [ i ] = R * np.exp( 1j * 2 * np.pi / M * i )
            
        self.set_symbols( symbols )
        self.set_gray_bits( self.m )
        
    def fphi(self, phis):
        rS = self.SNRS
        v = np.arange(0, self.vmax, self.vmax / self.Nphi)
        f = np.zeros(phis.size)
        
        for i, phi in enumerate(phis):
            g = v * np.exp( -0.5 * ( v - np.sqrt(2*rS) * np.cos(phi) ) ** 2.0 )
            self.v = v
            self.g = g
            f[i] = 1/(2*np.pi) * np.exp( -rS*np.sin(phi) ** 2.0) * np.trapz(g,v)
      
        return f
    
    def ser(self, Npoints = 100):
        
        phi = np.arange(-np.pi / self.M, np.pi / self.M, 2 * np.pi / self.M / Npoints)
        fphi = self.fphi(phi)
        return 1 - np.trapz(fphi, phi)
    
    def ber(self, Npoints = 1000):
        return self.ser(Npoints = Npoints) / self.m
#---pskconstellation2
        
class digital_signal(signal):
    
    def __init__(self, TS = 1e-6, samples_per_symbol = 10, 
                 tinitial = 0, tguard = 0.0, constellation = None,
                 fcarrier = 0, phi0 = 0):
    
        super().__init__()
        self.TS = TS
        self.samples_per_symbol = samples_per_symbol
        self.tinitial = tinitial
        self.tguard = tguard
        self.constellation =  constellation
        self.fcarrier = fcarrier
        self.phi0 = phi0
        
    def set_constellation(self, constellation):
        self.constellation = constellation
        
    def set_input_bits(self, bits):
        self.input_bits = bits        
       
#---modulatefromsymbols1
    def modulate_from_symbols( self, symbols ):
        
        self.Tmin = self.tinitial - self.tguard
        self.Tmax = self.Tmin + symbols.size * self.TS + 2 * self.tguard
        self.Dt = self.TS / self.samples_per_symbol
        self.t = np.arange(self.Tmin, self.Tmax, self.Dt)
        self.samples = np.zeros( self.t.size )
        self.symbols = symbols
        self.N = self.t.size
        
        i = np.floor( (self.t - self.tinitial) / self.TS).astype(int)
        j = np.where( np.logical_and(i >= 0, i < symbols.size ) )
        
        phase = 2 * np.pi * self.fcarrier * self.t[j] + self.phi0

        self.samples[j] = np.real(symbols[ i[j] ]) * np.cos( phase ) \
                        - np.imag(symbols[ i[j] ]) * np.sin( phase )
        self.phase = phase
        
#---modulatefromsymbols2
        
    def modulate_from_bits( self, bits, constellation = None):
        
        if constellation is not None:
            self.set_constellation(constellation)
        
        self.set_input_bits( bits )
        samples = self.constellation.bits_to_symbols( bits )
        self.modulate_from_symbols( samples )

class white_noise(signal):    

    def __init__(self, N0 = None, B = None, sigma2 = None, t = None, Nsamples = None):
        
        if sigma2 is None:
            sigma2 = N0 * B
        
        if t is not None:
            Nsamples = t.size
        
        samples = np.sqrt(sigma2) * np.random.randn( Nsamples )
        
        super().__init__(t = t, samples = samples)        
        
class system:
    
    def __init__(self, input_signal = None, transfer_function = None):
        self.input_signal = input_signal
        self.output_signal = None
        self.transfer_function = transfer_function
        
    def set_input(self, input_signal):
        self.input_signal = input_signal
        
    def set_transfer_function(self, transfer_function):
        self.transfer_function = transfer_function
        
    def set_output(self, output_signal):
        self.output_signal = output_signal
        
    def calc_transfer_function(self):
        if callable(self.transfer_function):
            f = self.input_signal.set_frequency_axis()
            self.transfer_samples = self.transfer_function(f)
            
    def apply(self):
        self.calc_transfer_function()
        self.input_signal.calc_spectrum()
        self.output_signal = deepcopy( self.input_signal )
        self.output_signal.spec = self.output_signal.spec * self.transfer_samples
        self.output_signal.calc_invspectrum()                          
        
    def get_input(self):
        return self.input_signal
    
    def get_transfer_function(self):
        return self.transfer_function
    
    def get_output(self):
        return self.output_signal

#---montecarlo1
class monte_carlo:
    def __init__(self, max_iterations = 1000, generate = None, 
                       apply = None, measure = None, report_step = 100,
                       report = False, keep_realizations = False):
        
        self.max_iterations = max_iterations
        self.report_step = report_step
        self.report = report
        self.keep_realizations = keep_realizations
        self.realizations = []
        self.ci = 0
    
    def report_iteration(self, i):
        print('iteration %d / %d' %(i, self.max_iterations) )
        
    def execute(self):
        for i in range(self.max_iterations):
            if self.report and ( np.mod(i, self.report_step) == 0 ):
                self.report_iteration(i)
                
            self.generate()
            self.apply()
            self.measure()
       
            if self.keep_realizations:
                self.append_to_realizations()
                
            if self.terminate():
                self.termination_condition = True
                self.iterations_performed = i
                return
            
            self.ci += 1
        
        self.termination_condition = False
        self.iterations_performed = i
        
    def plot_realizations(self):
        plt.plot(np.real(self.realizations), np.imag(self.realizations), 'o')

#---histstaff    

    def make_histogram(self, key = None, **kwargs ):
        self.data = np.array( [key(x) for x in self.realizations] )       
        self.histogram = np.histogram( self.data, **kwargs )
    
    def plot_histogram(self, figure_no = None, normalize_by = 1):
        
        if figure_no is None:
            plt.figure()
        else:
            plt.figure(figure_no)
        
        plt.hist(self.data / normalize_by)
    
#---montecarlo2

#---pamsimulation1                       
class pam_simulation(monte_carlo):
    
    def __init__(self, max_iterations = 1000, M = 16, SNRbdB = 10, report_step = 10, max_symbol_errors = 100):
        
        super().__init__(max_iterations = max_iterations, report_step = report_step )
        self.M = M
        self.m = np.log2(M).astype(int)
        self.SNRbdB = SNRbdB
        self.SNRb = 10 ** (SNRbdB / 10)
        self.constellation = pam_constellation(M)
        self.sigma = np.sqrt( (M ** 2.0 - 1) / ( 6 * self.SNRb *np.log2(M) ) ) 
        self.symbol_errors = 0
        self.bit_errors = 0
        self.max_symbol_errors = max_symbol_errors
        
    def generate(self):
        bits = random_bits( self.m )
        symbol = self.constellation.bits_to_symbols( bits )
        noise = self.sigma * np.random.randn( 1 )
        
        self.symbol = symbol
        self.input_bits = bits
        self.noise = noise
            
    def apply(self):
        output = self.symbol + self.noise
        [self.decoded_symbol, self.decoded_bits, _ ] = self.constellation.decode( output )
                
    def measure(self):
        self.bit_errors += np.sum( np.abs(self.decoded_bits - self.input_bits) ).astype(int)
        self.symbol_errors += int(self.symbol != self.decoded_symbol)
    
    def terminate(self):
        return self.symbol_errors >= self.max_symbol_errors
#---pamsimulation2        

#---psksimulation1
class psk_simulation(monte_carlo):
    
    def __init__(self, M = 16, SNRbdB = 10, max_symbol_errors = 100, **kwargs):

        super().__init__( **kwargs )
        self.M = M
        self.m = np.log2(M).astype(int)
        self.SNRbdB = SNRbdB
        self.SNRb = 10 ** (SNRbdB / 10)
        self.constellation = psk_constellation(M, SNRbdB = SNRbdB)
        self.N0 = 1 / self.SNRb / np.log2(M)
        self.sigma = np.sqrt(self.N0 / 2)
        self.symbol_errors = 0
        self.bit_errors = 0
        self.max_symbol_errors = max_symbol_errors 
        self.realizations = np.zeros( self.max_iterations, dtype = complex )        

    def generate(self):
        bits = random_bits( self.m )
        self.symbol = self.constellation.bits_to_symbols( bits )
        self.noise = self.sigma * np.random.randn( 1 ) + 1j * self.sigma * np.random.randn(1)
        self.input_bits = bits
        
    def apply(self):
        self.output = self.symbol + self.noise
        [self.decoded_symbol, self.decoded_bits, _ ] = self.constellation.decode( self.output )
        
    def measure(self):
        self.bit_errors += np.sum( np.abs(self.decoded_bits - self.input_bits) ).astype(int)
        self.symbol_errors += int(self.symbol != self.decoded_symbol)
#---pskmeasure
        
    def terminate(self):
        return self.symbol_errors >= self.max_symbol_errors
    
    def append_to_realizations(self):
        self.realizations[self.ci] = self.output[0]
    
    def report_iteration(self,i):
        if i != 0:            
            ser = self.symbol_errors / i
            print('iteration %d / %d, symbol errors = %d, SER = %e' %(i, self.max_iterations, self.symbol_errors, ser) )
         
    def plot_constellation(self):
        plt.figure()
        self.plot_realizations()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('M= %d, SNRb = %2.1f dB' %(self.M, self.SNRbdB))
        plt.axis('equal')
#---psksimulation2

#---binarysequence1
class binary_sequence_simulation(monte_carlo):
    
    def __init__(self, sze = 100, p1 = 0.5, **kwargs):
        super().__init__( keep_realizations = True, **kwargs )
        self.sze = sze
        self.p1 = p1
        self.p0 = 1 - p1
        
    def generate(self):
        self.bits = (np.random.rand( self.sze ) <= self.p1).astype(int)
        
    def apply(self):
        pass
    
    def measure(self):
        self.no_ones = np.sum( self. bits )
        self.no_zeroes = self.sze - self.no_ones
        
    def terminate(self):
        return False
    
    def append_to_realizations(self):
        self.realizations.append({'ones' : self.no_ones,
                                  'zeroes' : self.no_zeroes})
#---binarysequence2
    
        
#---multiplymodulo2
def multiply_modulo2(v, G):
    return (v.dot(G) % 2).astype(int)

def add_str_modulo2(s1, s2):
    a1 = str_to_array(s1)
    a2 = str_to_array(s2)
    return array_to_str( np.bitwise_xor(a1, a2) )

def multiply_str_modulo2(v, G):
    v = str_to_array(v)
    return array_to_str( multiply_modulo2( v, G ) )
            
class block_code:
    
    def __init__(self, k, n, G = None, compute_codewords = False):
        self.k = int(k)
        self.n = int(n)
        self.mw_strs = []
        self.mw = np.zeros([2 ** k, k], dtype = int)
        
        self.G = G.astype(int)
        if compute_codewords:
            self.set_messages()
            self.set_codewords()

    def set_generator_matrix(self, G):
        self.G = G.asytpe(int)
    
    def set_messages(self):
        
        for i in range(0, 2 ** self.k):
            bstr = np.binary_repr(i, width = self.k)
            self.mw_strs.append(bstr)
            self.mw[i, :] = str_to_array(bstr)

#---blockcodemiddle           
    def set_codewords(self):
        self.cw = np.zeros([2 ** self.k, self.n], dtype = int)
        self.cw_str = []
        
        for i in range(0, 2 ** self.k) :
            cw = multiply_modulo2(self.mw[i , :] , self.G)
            self.cw[i, :] = cw.astype(int)
        
    def calc_weights(self):
        self.weights = np.zeros( 2**self.k )
        self.min_dist = np.inf
        
        for i, cw in enumerate(self.cw):
            self.weights[i] = np.sum(cw)
            if self.weights[i] < self.min_dist and self.weights[i] > 0:
                self.min_dist = self.weights[i]
            
    def min_weight( self ):
        i = np.where( self.weights > 0 )
        return np.min( self.weights[i] )    
    
    def calc_distances(self):
        self.distances = np.zeros([2**self.k, 2**self.k])
        self.min_dist = np.inf 
        
        for p in range(0, 2**self.k):
            for q in range(0, 2**self.k):
                
                if ( p != q ):
                    d = np.sum(np.logical_xor(self.cw[p], self.cw[q]))
                    if d < self.min_dist:
                        self.min_dist = d
                    
                    self.distances[p,q] = d 
    
    def min_distance( self ):
        return self.min_dist

    def print(self):
        for i, mw in enumerate(self.mw):
            cw = self.cw[i]
            print(mw, '-->', cw)

#---buildstandardarray1            
    def build_standard_array(self):
        
        combinations = []
        print('Building combinations')
        
        for i in range(0, 2 ** self.n):
            bstr = np.binary_repr(i, width = self.n)
            combinations.append(bstr)

        print('Building codeword lists')        
        cw_strs = []
        for cw in self.cw:
            bstr = array_to_str(cw)
            cw_strs.append(bstr)
            
        print('Removing codewords from combination pool')        
        combinations = [x for x in combinations if x not in cw_strs]
        combinations = sorted(combinations, key = lambda x: x.count('1'))
                
        print('Setting first standard array row')
        standard_array = deepcopy(cw_strs)
        
        i = 2
        while len(combinations) > 0:
            print('Adding row %i , there are now %d elements in the array' %(i, len(standard_array)))
            driver = combinations[0] 
            print('driver now is %s' %driver)
            new_elements = [ add_str_modulo2(driver,x) for x in cw_strs]
            print('New row to be added :%s' % new_elements)
            standard_array.extend( new_elements )            
            combinations = [x for x in combinations if x not in new_elements]
            i += 1
            print('\n')

#---buildstandardarray2            
        print('Rearranging array')
        self.standard_array = []
        self.drivers = []
        self.syndromes = {}
        
        for i in range(0, len(standard_array), len(cw_strs) ):
            self.standard_array.append( standard_array[i:i+len(cw_strs)] )
            driver = standard_array[i]
            self.drivers.append( driver )
            syndrome = multiply_str_modulo2( driver, self.Ht )
            self.syndromes[syndrome] = driver
            
    def print_standard_array(self):
        for row in self.standard_array:
            print(row)            
#---blockcodeend1           
            
#---systematiccode1
class systematic_code( block_code ):
    
    def __init__(self, P = None, compute_codewords = False):
    
        k, ny = P.shape
        n = k + ny
        I = np.identity(k, dtype = int)
        G = np.concatenate( (I,P), axis = 1)            
        self.P = P.astype(int)
        super().__init__(k, n, G = G, compute_codewords = compute_codewords)
        self.set_parity_matrix()       
            
    def set_parity_matrix(self):
        I = np.identity(self.n - self.k, dtype = int)
        self.H = np.concatenate( (np.matrix.transpose(self.P), I), axis = 1 )
        self.Ht = np.matrix.transpose(self.H)
#---systematiccode2
   
#---hammingcode1     
class hamming_code( systematic_code ):
    
    def __init__(self, m = 3, compute_codewords = False):
        k = int(2 ** m - m - 1)
        n = k + m
        Pt = np.zeros( [n-k, k] )
        c = 0
        
        for i in range(1, 2 ** m):
            bstr = np.binary_repr(i, m)
            b = str_to_array(bstr)
            if sum(b) >= 2:
                Pt[:, c] = b
                c += 1
            
        P = np.transpose(Pt)
        super().__init__( P = P, compute_codewords = compute_codewords )
#---hammingcode2     

            
#---lzw1
class lzwcompressor:
    
    def __init__(self):
        self.binreps = []          # list to hold the binary representations of the messages
        self.decreps = []          # list to hold the decimal representations
        self.msgs = []             # list to hold the messages
        self.max_so_far = 127
        self.bin_msg = ''
        
        self.create_initial_dict()
        
    # pre-encode initial phrases in dictionary
    def encode_ascii(self,ir):
        for i in ir:
            binrep = bin(i)[2:].zfill(7)             
            self.decreps.append(i)                    
            self.binreps.append(binrep)                          
            self.msgs.append(chr( i ) )               
    
    # output code corresponding to w. binmsg is the compressed binary message so far
    def output_code(self, w):
        i=self.msgs.index(w)                       
        binrep=self.binreps[i]                     
        self.bin_msg=binrep+self.bin_msg           
    
    # add entry to dictionary
    def add_entry(self,wc):
        self.max_so_far=self.max_so_far+1           
        binrep=bin(self.max_so_far)[2:]          
        self.msgs.append(wc)                      
        self.decreps.append(self.max_so_far)    
        self.binreps.append(binrep)              
       
#---lzw2
         
    def contains(self,wc):
        return wc in self.msgs
        
    def compress(self):
        # phrase read so far
        w = ''

        # message to transmit    
        self.bin_msg=''

        # for every character in S
        for c in self.source:
            if self.contains ( w + c ):
                w += c
            else:
                # output code for w
                self.output_code(w)
                
                # add entry for w+c
                self.add_entry( w + c )
                w = c
    
    def read_from_file(self,filename):
        with open(filename,'r') as f:
            self.source = f.read()
            
    def create_initial_dict(self):
        ir=range(1,128)
        self.encode_ascii(ir)
    
    def compression_level(self):
        return len(self.bin_msg) / 7 / len(self.source)
#---lzw3