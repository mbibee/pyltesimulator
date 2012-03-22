#@+leo-ver=5-thin
#@+node:Michael.20120321090100.1528: * @thin ./Simulation/PBCH/pbch_gen_detect.py
#@+others
#@+node:Michael.20120321090100.1527: ** source
from scipy.signal import *
from numpy import *
import matplotlib.pyplot as plt

# time scale is in 1 s
T_s = 1.0/30720/1000 # in s

# configuration for SSS
l = 0
N_maxDL_RB = 110
N_DL_RB = 110
N_RB_sc = 12
N_DL_CP = 0 # normal DL CP
AP_num = 2
CP_DL_type = 0
N_DL_symb = 7
delta_f = 15000
subframe = 0
N_cell_ID = 0
f_0 = (2620+0.1*(2620-2750))*1000*1000  # in Hz

if N_DL_CP==0 and delta_f==15000:
    if l==0:
        N_CP_l = 160
    else:
        N_CP_l = 144
elif N_DL_CP==1:    # extended CP
    if delta_f==15000:
        N_CP_l = 512
    else:   # delta_f == 7500
        N_CP_l = 1024
if delta_f==15000:
    N = 2048
else:   # delta_f == 7500
    N = 4096

t = arange(0, (N_CP_l+N)*T_s, T_s)

def find_max( a_list ):
    m = max(a_list)
    for i in arange(len(a_list)):
        if a_list[i] == m:
            return (i, m)

def find_min( a_array ):
    x, y = 0, 0
    for i in arange(len(a_array)):
        if a_array[i] < y:
            x, y = i, a_array[i]
    return (x,y)

def find_abs_max( a_array ):
    m = max(abs(a_array))
    for i in arange(len(a_array)):
        if abs(a_array[i]) == m:
            return (i, m)
            
#@+others
#@+node:Michael.20120321090100.1529: *3* 01. PBCH symbol array in one OFDM symbol
def test_PBCH_symbol_array_in_one_OFDM_symbol():
    
    layer_mapping_scheme = 'single_antenna'
    num_of_layers = 1
    precoding_scheme = 'single_antenna'
    AP_num = 1
    codebook_index = 0
    
    
    a = array( [1]*24 )
    e = BCH_channel_process(a, AP_num, CP_DL_type)
    n_s = 1
    CSRS_RE_tuple = get_CSRS_REs_in_slot(n_s, 0, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 1, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 2, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 3, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    
    pbch_symbol_matrix = get_PBCH_symbol_matrix(e, N_cell_ID, layer_mapping_scheme, num_of_layers, precoding_scheme, AP_num, codebook_index, N_DL_RB, N_RB_sc, CSRS_RE_tuple)
    pbch_symbol_matrix_ap0 = pbch_symbol_matrix[0]
    for l in range(4):
        #legend_list = list()
        plt.title("PBCH symbol array in l=%s for AP=0"%l)
        plt.plot(abs(pbch_symbol_matrix_ap0[l]))
        plt.xlabel("k (subcarrier index)")
        plt.ylabel("Magnitude")
        plt.show()
#@+node:Michael.20120321090100.1538: *3* 02. PBCH Uu signal
def test_PBCH_Uu_signal():
    
    layer_mapping_scheme = 'single_antenna'
    num_of_layers = 1
    precoding_scheme = 'single_antenna'
    AP_num = 1
    codebook_index = 0
    
    
    a = array( [1]*24 )
    e = BCH_channel_process(a, AP_num, CP_DL_type)
    n_s = 1
    CSRS_RE_tuple = get_CSRS_REs_in_slot(n_s, 0, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 1, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 2, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    CSRS_RE_tuple += get_CSRS_REs_in_slot(n_s, 3, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    
    pbch_symbol_matrix = get_PBCH_symbol_matrix(e, N_cell_ID, layer_mapping_scheme, num_of_layers, precoding_scheme, AP_num, codebook_index, N_DL_RB, N_RB_sc, CSRS_RE_tuple)
    pbch_symbol_matrix_ap0 = pbch_symbol_matrix[0]
    
    pbch_Uu_signal = array( [0.0] * (160+144*3+2048*4) )
    t0 = arange(0, (160+2048)*T_s, T_s)
    t1 = arange(0, (144+2048)*T_s, T_s)
    for l in range(4):
        if l==0:
            pbch_re_array = pbch_symbol_matrix_ap0[l]
            pbch_baseband_IQ = ofdm_baseband_IQ_signal_generate(pbch_re_array, l, N_DL_RB, N_RB_sc, N_DL_CP, delta_f)
            tmp_pbch_Uu_signal = downlink_modulate(pbch_baseband_IQ, t0, f_0)
            pbch_Uu_signal[:(160+2048)] = tmp_pbch_Uu_signal
        else:
            pbch_re_array = pbch_symbol_matrix_ap0[l]
            pbch_baseband_IQ = ofdm_baseband_IQ_signal_generate(pbch_re_array, l, N_DL_RB, N_RB_sc, N_DL_CP, delta_f)
            tmp_pbch_Uu_signal = downlink_modulate(pbch_baseband_IQ, t1, f_0)
            pbch_Uu_signal[(160+2048)+(144+2048)*(l-1):(160+2048)+(144+2048)*l] = tmp_pbch_Uu_signal
    new_t = arange(0, (160+144*3+2048*4)*T_s, T_s)
    plt.plot(new_t*1000, pbch_Uu_signal)
    plt.title('PBCH (only) Uu signal for ap=0')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal level')
    #plt.legend(legend_list)
    #plt.axis( [-0.01, 0.075, -0.1, 14] )
    plt.show()
#@+node:michael.20120305092148.1277: *3* 36.211
#@+others
#@+node:michael.20120305092148.1295: *4* 6 Downlink
#@+others
#@+node:michael.20120305092148.1304: *5* 5.07.2 Preamble seq gen
# u_th root Zadoff-Chu sequence
def x_u(u, n, N_ZC):
    if n>=0 and n<=N_ZC:
        return exp(-1j*pi*u*n*(n+1)/N_ZC)
    else:
        return 0

# u_th root Z-C sequence with v N_cs cyclic shift
def x_uv(u, v, n, N_ZC):
    pass
#@+node:michael.20120305092148.1278: *5* 6.02.4 REGs
def get_REG_in_RB_symbol(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type, optimize=True):
    '''
    get_REG_in_RB_symbol(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type, optimize=True): tuple of REGs
    Return REGs in the specific PRB of symbol l.
    n_PRB: PRB index
    N_RB_sc: number of subcarriers in one RB
    l: symbol index
    CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
    DL_CP_type: 0 for normal CP, 1 for extended CP
    optimize: boolean. whether to use optimized code for speed.
    '''
    if optimize:
        result = get_REG_in_RB_symbol_opti(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type)
    else:
        result = get_REG_in_RB_symbol_standard(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type)
    return result
    
def get_REG_in_RB_symbol_standard(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type):
    '''
    get_REG_in_RB_symbol(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type): tuple of REGs
    Return REGs in the specific PRB of symbol l.
    n_PRB: PRB index
    N_RB_sc: number of subcarriers in one RB
    l: symbol index
    CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
    DL_CP_type: 0 for normal CP, 1 for extended CP
    '''
    k_0 = n_PRB * N_RB_sc
    if l == 0:
        REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l), (k_0+4,l), (k_0+5,l) ),
                        ( (k_0+6,l), (k_0+7,l), (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l) )
                    )
    elif l ==1:
        if CSRS_AP_num == 4:
            REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l), (k_0+4,l), (k_0+5,l) ),
                        ( (k_0+6,l), (k_0+7,l), (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l) )
                    )
        else:   # it is the same for 1/2 CSRS APs, and also for calculating REG 2 CSRS APs shall be assumed for single AP configuration
            REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l)    ),
                            (   (k_0+4,l), (k_0+5,l), (k_0+6,l), (k_0+7,l)    ),
                            (   (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l)    )
                        )
    elif l ==2:
        REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l)    ),
                        (   (k_0+4,l), (k_0+5,l), (k_0+6,l), (k_0+7,l)    ),
                        (   (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l)    )
                    )
    elif l ==3:
        if DL_CP_type ==0:
            REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l)    ),
                            (   (k_0+4,l), (k_0+5,l), (k_0+6,l), (k_0+7,l)    ),
                            (   (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l)    )
                        )
        else:   # extended DL CP
            REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l), (k_0+4,l), (k_0+5,l) ),
                            ( (k_0+6,l), (k_0+7,l), (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l) )
                        )
    return REGs

def get_REG_in_RB_symbol_opti(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type):
    '''
    Return REGs in the specific PRB of symbol l, optimized version.
    n_PRB: PRB index
    N_RB_sc: number of subcarriers in one RB
    l: symbol index
    CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
    DL_CP_type: 0 for normal CP, 1 for extended CP
    '''
    k_0 = n_PRB * N_RB_sc
    if (l== 0) or (l==1 and CSRS_AP_num==4) or (l==3 and DL_CP_type==1):
        REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l), (k_0+4,l), (k_0+5,l) ),
                        (   (k_0+6,l), (k_0+7,l), (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l) )
                    )
    else:
        REGs = (    (   (k_0,l), (k_0+1,l), (k_0+2,l), (k_0+3,l)    ),
                        (   (k_0+4,l), (k_0+5,l), (k_0+6,l), (k_0+7,l)    ),
                        (   (k_0+8,l), (k_0+9,l), (k_0+10,l), (k_0+11,l)    )
                    )
    return REGs

#@+others
#@+node:Michael.20120320091224.1504: *6* more helper not explicitly in protocol
def get_REG_from_kl(num_of_ap, CP_DL_type, k, l):
    '''
    input:
        num_of_ap: number of antenna ports used for Cell Specific Reference Signal
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
        k: subcarrier index k
        l: OFDM symbol index l in one slot
    output:
        tuple representation of a REG containing (k,l), including the REs used for CSRS.
    '''
    result = None
    if l == 0:
        result = ( (k/6*6,l), (k/6*6+1,l), (k/6*6+2,l), (k/6*6+3,l), (k/6*6+4,l), (k/6*6+5,l) )
    elif l == 1:
        if num_of_ap == 2:
            result = ( (k/4*4,l), (k/4*4+1,l), (k/4*4+2,l), (k/4*4+3,l) )
        elif num_of_ap ==4:
            result = ( (k/6*6,l), (k/6*6+1,l), (k/6*6+2,l), (k/6*6+3,l), (k/6*6+4,l), (k/6*6+5,l) )
    elif l == 2:
        result = ( (k/4*4,l), (k/4*4+1,l), (k/4*4+2,l), (k/4*4+3,l) )
    elif l == 3:
            if CP_DL_type == 0:    # for nomal CP
                result = ( (k/4*4,l), (k/4*4+1,l), (k/4*4+2,l), (k/4*4+3,l) )
            else:
                result = ( (k/6*6,l), (k/6*6+1,l), (k/6*6+2,l), (k/6*6+3,l), (k/6*6+4,l), (k/6*6+5,l) )
    return result

#@-others
#@+node:Michael.20120319125504.1471: *5* 6.03.3 Layer mapping
#@+others
#@+node:Michael.20120319125504.1472: *6* 6.3.3.1 Layer mapping for transmission on a single antenna port
def layer_map_single_antenna_port(x):
    '''
    input:
        array of symbols
    output:
        array of arrays of the layer mapped symbols. e.g. output[0] is the array for layer 0
    '''
    v = 1
    M_layer_symb = len(x)
    return array( [x] )
#@+node:Michael.20120319125504.1473: *6* 6.3.3.2 Layer mapping for spatial multiplexing
def layer_map_spatial_multiplex(codewords, v):
    '''
    input:
        codewords: array of codewords. codewords[0] is the first codeword, and codewords[1] is the second.
        v: number of layers, integer (1..4)
    output:
        layer mapped matrix: layer_mapped_matrix[0] is the symbol list for the first layer
    '''
    result = None
    num_of_cw = len(codewords)
    if v==1 and num_of_cw==1:
        #M_layer_symb = len(codewords[0])
        result = array( [codewords[0]] )
        
    elif v==2 and num_of_cw==2:
        #M_layer_symb = len(codewords[0])
        assert( len(codewords[0]) == len(codewords[1]) )
        result = array(codewords)
        
    elif v==2 and num_of_cw==1:
        M_layer_symb = len(codewords[0])/2
        x0 = array( [0.0+0.0*1j] * M_layer_symb )
        x1 = array( [0.0+0.0*1j] * M_layer_symb )
        for i in range(M_layer_symb):
            x0[i] = codewords[0][2*i]
            x1[i] = codewords[0][2*i+1]
        result = array( [x0, x1] )
        
    elif v==3 and num_of_cw==2:
        assert( len(codewords[0]) == len(codewords[1])/2 )
        M_layer_symb = len(codewords[0])
        x0 = array( [0.0+0.0*1j] * M_layer_symb )
        x1 = array( [0.0+0.0*1j] * M_layer_symb )
        x2 = array( [0.0+0.0*1j] * M_layer_symb )
        for i in range(M_layer_symb):
            x0[i] =codewords[0][i]
            x1[i] = codewords[1][2*i]
            x2[i] = codewords[1][2*i+1]
        result = array( [x0, x1, x2] )
    
    elif v==4 and num_of_cw==2:
        assert( len(codewords[0]) == len(codewords[1]) )
        M_layer_symb = len(codewords[0])/2
        x0 = array( [0.0+0.0*1j] * M_layer_symb )
        x1 = array( [0.0+0.0*1j] * M_layer_symb )
        x2 = array( [0.0+0.0*1j] * M_layer_symb )
        x3 = array( [0.0+0.0*1j] * M_layer_symb )
        for i in range(M_layer_symb):
            x0[i] = codewords[0][2*i]
            x1[i] = codewords[0][2*i+1]
            x2[i] = codewords[1][2*i]
            x3[i] = codewords[1][2*i+1]
        result = array( [x0, x1, x2, x3] )
    
    return result
#@+node:Michael.20120319125504.1475: *6* 6.3.3.3 Layer mapping for transmit diversity
def layer_map_transmit_diversity(codewords, v):
    '''
    input:
        codewords: array of codewords. codewords[0] is the first codeword, and codewords[1] is the second.
        v: number of layers, integer 2 or 4
    output:
        layer mapped matrix: layer_mapped_matrix[0] is the symbol list for the first layer
    '''
    result = None
    num_of_cw = len(codewords)
    assert( num_of_cw==1 )
        
    if v==2:
        M_layer_symb = len(codewords[0])/2
        x0 = array( [0.0+0.0*1j] * M_layer_symb )
        x1 = array( [0.0+0.0*1j] * M_layer_symb )
        for i in range(M_layer_symb):
            x0[i] = codewords[0][2*i]
            x1[i] = codewords[0][2*i+1]
        result = array( [x0, x1] )
    
    elif v==4:
        if len(codewords[0]) % 4 == 0:
            M_layer_symb = len(codewords[0])/4
            tmp_cw = codewords[0]
        else:
            M_layer_symb = (len(codewords[0])+2)/4
            tmp_cw = array( [0.0+0.0*1j] * (len(codewords[0]) + 2) )
            tmp_cw[:len(codewords[0])] = codewords[0]
        x0 = array( [0.0+0.0*1j] * M_layer_symb )
        x1 = array( [0.0+0.0*1j] * M_layer_symb )
        x2 = array( [0.0+0.0*1j] * M_layer_symb )
        x3 = array( [0.0+0.0*1j] * M_layer_symb )
        for i in range(M_layer_symb):
            x0[i] = tmp_cw[4*i]
            x1[i] = tmp_cw[4*i+1]
            x2[i] = tmp_cw[4*i+2]
            x3[i] = tmp_cw[4*i+3]
        result = array( [x0, x1, x2, x3] )
    
    return result
#@-others
#@+node:Michael.20120319125504.1476: *5* 6.03.4 Precoding
#@+others
#@+node:Michael.20120319125504.1477: *6* 6.3.4.1 Precoding for transmission on a single antenna port
def precode_single_antenna(layer_mapped_matrix):
    '''
    input:
        layer_mapped_matrix: matrix result of layer mapping
    output:
        precoded matrix. precoded_matrix[0] is the result for antenna port 0.
    '''
    #M_layer_symb = len(layer_mapped_matrix[0])
    #M_ap_symb = M_layer_symb
    return layer_mapped_matrix
#@+node:Michael.20120319125504.1478: *6* 6.3.4.2 Precoding for spatial multiplexing using antenna ports with CSRS
#@+others
#@+node:Michael.20120319125504.1479: *7* 6.3.4.2.1 Precoding without CDD
#@+node:Michael.20120319125504.1480: *7* 6.3.4.2.3 Codebook for precoding
def get_codebook_for_precoding( ap_num, codebook_index, v ):
    '''
    input:
        ap_num: number of transmiting antennas
        codebook_index: Codebook index
        v: number of layers
    output:
        an array representing selected codebook
    '''
    lte_assert(ap_num in (2,4), 'Precoding codebook is only valid for transmission on two or four antenna ports.' )
    result = None
    if ap_num==2:
        lte_assert(codebook_index in (0,1,2,3), 'Codebook index (%s) out of range. Please refer to 36.211 table 6.3.4.2.3-1 in section 6.3.4.2.3.'%codebook_index)
        if v==1:
            x, y =( (1,1), (1,-1), (1,1j), (1,-1j) )[codebook_index]
            result = 1/sqrt(2) * array( [[x], [y]] )
        elif v==2:
            lte_assert(codebook_index!=3, "Codebook index (%s) for %s layers isn't defined. Please refer to 36.211 table 6.3.4.2.3-1 in section 6.3.4.2.3."%(codebook_index, v))
            x1, x2, y1, y2 = ( (1/sqrt(2),0,0,1/sqrt(2)), (0.5,0.5,0.5,-0.5), (0.5,0.5,0.5j,-0.5j) )[codebook_index]
            result = array( [[x1,x2], [y1,y2]] )
    else:   # ap_num==4
        raise LteException("36.211 6.3.4.2.3 codebook selection for 4 transmission antenna. Need to be added.")
#@-others
#@+node:Michael.20120320091224.1502: *6* 6.3.4.3 Precoding for transmit diversity
def precode_transmit_diversity(layer_mapped_matrix, num_of_ap):
    '''
    input:
        layer_mapped_matrix: symbol matrix after layer mapping. e.g. layer_mapped_matrix[0] is the symbol array for layer 0.
        num_of_ap: number of transmitting antenna.
    output:
        precoded matrix. e.g. output[0] is the symbol array for ap 0.
    '''
    num_of_layers = len(layer_mapped_matrix)
    lte_assert(num_of_ap==num_of_layers, "For transmit diversity precoding, number of transmitting antenna must be equal to number of layers, but currently num_of_ap=%s and num_of_layers=%s"%(num_of_ap, num_of_layers))
    lte_assert(num_of_ap in (2,4), "For 'transmit_diversity' precoding scheme, number of transmission antenna must be 2 or 4. Current number of transmitting antenna is %s"%num_of_ap)
    if num_of_ap==2:
        M_layer_symb = len(layer_mapped_matrix[0])
        M_ap_symb = 2*M_layer_symb
        #codebook = 1/sqrt(2) * array([ [1,0,1j,0], [0,-1,0,1j], [0,1,0,1j], [1,0,-1j,0] ])
        y = array([ [0.0+0.0*1j]*M_ap_symb, [0.0+0.0*1j]*M_ap_symb ])
        x = layer_mapped_matrix
        for i in range(M_layer_symb):
            tmp_x = array([real(x[0][i]), real(x[1][i]), imag(x[0][i]), imag(x[1][i])])
            y[0][2*i] = 1/sqrt(2) * sum(array([1,0,1j,0])) * tmp_x
            y[1][2*i] = 1/sqrt(2) * sum(array([0,-1,0,1j])) * tmp_x
            y[0][2*i+1] = 1/sqrt(2) * sum(array([0,1,0,1j])) * tmp_x
            y[1][2*i+1] = 1/sqrt(2) * sum(array([1,0,-1j,0])) * tmp_x
    else:
        # num_of_ap==4
        M_layer_symb = len(layer_mapped_matrix[0])
        if M_layer_symb%4==0:
            M_ap_symb = M_layer_symb * 4
        else:
            M_ap_symb = M_layer_symb * 4 -2
        y = array([ [0.0+0.0*1j]*M_ap_symb, [0.0+0.0*1j]*M_ap_symb, [0.0+0.0*1j]*M_ap_symb, [0.0+0.0*1j]*M_ap_symb ])
        x = layer_mapped_matrix
        for i in range(M_layer_symb):
            tmp_x = array([real(x[0][i]), real(x[1][i]), real(x[2][i]), real(x[3][i]), imag(x[0][i]), imag(x[1][i]), imag(x[2][i]), imag(x[3][i])])
            y[0][4*i] = 1/sqrt(2) * sum(array([1,0,0,0,1j,0,0,0])) * tmp_x
            y[1][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[2][4*i] = 1/sqrt(2) * sum(array([0,-1,0,0,0,1j,0,0])) * tmp_x
            y[3][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[0][4*i+1] = 1/sqrt(2) * sum(array([0,1,0,0,0,1j,0,0])) * tmp_x
            y[1][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[2][4*i+1] = 1/sqrt(2) * sum(array([1,0,0,0,-1j,0,0,0])) * tmp_x
            y[3][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[0][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[1][4*i+2] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,1j,0])) * tmp_x
            y[2][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[3][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,-1,0,0,0,1j])) * tmp_x
            y[0][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[1][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,1,0,0,0,1j])) * tmp_x
            y[2][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            y[3][4*i+3] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,-1j,0])) * tmp_x
    return y
#@-others
#@+node:Michael.20120321090100.1521: *5* 6.06 PBCH
#@+others
#@+node:Michael.20120321090100.1522: *6* 6.6.1 Scrambling
def PBCH_scramble(b, N_cell_ID):
    '''
    input:
        b: array of PBCH bits
        N_cell_ID: cell ID
    output: tuple of integer, the scrambled bits
    '''
    lte_assert(len(b) in (1920,1728), "PBCH bits has to be 1920 for normal CP or 1728 for extended CP, but current bit array length is %s"%len(b))
    c_init = N_cell_ID
    b_ = [0] * len(b)
    for i in range(len(b)):
        b_[i] = (b[i] + c(c_init,i))%2
    return tuple(b_)
#@+node:Michael.20120321090100.1523: *6* 6.6.2 Modulation
def PBCH_modulate(b_):
    '''
    input:
        b_: scrambled PBCH but sequence
    output:
        QPSK modulated symbol array (complex number)
    '''
    d = array([0.0 + 1j*0.0] * (len(b_)/2))
    for i in range(len(d)):
        d[i] = QPSK( (b_[2*i], b_[2*i+1]) )
    return d
#@+node:Michael.20120321090100.1524: *6* 6.6.3 Layer mapping and precoding
def PBCH_layer_map(d, layer_mapping_scheme, num_of_layers):
    '''
    input:
        b: array of QPSK modulated symbols
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
    output:
        layer mapped array of arrays for each layer. output[0] is the array for layer 0, and so on.
    '''
    lte_assert(layer_mapping_scheme in ('single_antenna', 'transmit_diversity'), "layer_mapping_scheme=%s is not valid for PBCH. It must be 'single_antenna' or 'transmit_diversity'."%(layer_mapping_scheme, ))
    if layer_mapping_scheme == 'single_antenna':
        lte_assert(num_of_layers==1, "For single antenna scheme of layer mapping, number of layers must be 1.")
        result = layer_map_single_antenna_port(d)
    else:
        lte_assert(num_of_layers in (2,4), "For transmit diversity scheme of layer mapping, number of layers must be either 2 or 4.")
        result = layer_map_transmit_diversity([d], num_of_layers)
    return result

def PBCH_precode(layer_mapped_matrix, precoding_scheme, num_of_ap, codebook_index):
    '''
    input:
        layer_mapped_matrix: matrix after PBCH layer mapping.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_ap: number of transmission antenna ports
        codebook_index: codebook index
    output:
        maxtrix for all transmission antenna ports. e.g. output[0] is the symbol array for ap_0.
    '''
    lte_assert(precoding_scheme in ('single_antenna', 'transmit_diversity'), "For PBCH precoding, the scheme must either be 'single_antenna' or 'transmit_diversity', but the current scheme is %s"%precoding_scheme)
    num_of_layers = len(layer_mapped_matrix)
    if precoding_scheme == 'single_antenna':
        lte_assert(num_of_ap==1, "For PBCH 'single_antenna' precoding scheme, num_of_ap must be 1, but currently it is %s."%num_of_ap)
        lte_assert(num_of_layers==1, "For PBCH 'single_antenna' precoding scheme, number of layers must be 1, but currently it is %s."%num_of_layers)
        result = precode_single_antenna(layer_mapped_matrix)
    else:
        # precoding_scheme == 'transmit_diversity'
        lte_assert(num_of_ap in (2,4), "For PBCH 'transmit_diversity' precoding scheme, number of transmission antenna must be 2 or 4. Current number of transmitting antenna is %s"%num_of_ap)
        lte_assert(num_of_ap==num_of_layers, "For transmit diversity precoding, number of transmitting antenna must be equal to number of layers, but currently num_of_ap=%s and num_of_layers=%s"%(num_of_ap, num_of_layers))
        result = precode_transmit_diversity(layer_mapped_matrix, num_of_ap)
    return result
#@+node:Michael.20120321090100.1525: *6* 6.6.4 Mapping to resource elements
def get_REs_for_PBCH_in_symbol(l, N_DL_RB, N_RB_sc, CSRS_REs):
    '''
    input:
        l: symbol index in one slot
        N_DL_RB: downlink Resource Block number in one slot
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_REs: all Cell Specific Reference Signal REs for 4 antenna, at least for the current symbol
    output:
        tuple of RE (k,l) that shall be used for PBCH
    '''
    lte_assert(l in (0,1,2,3), "For PBCH can only exist in the first 4 OFDM symbols, but the current symbol index is %s"%l)
    result = list()
    
    for k_ in range(72):
        k = N_DL_RB*N_RB_sc/2 - 36 + k_
        if (k,l) not in CSRS_REs:
            result.append( (k,l) )
    
    return tuple(result)


def get_REs_for_PBCH_in_slot(N_DL_RB, N_RB_sc, CSRS_REs):
    '''
    input:
        N_DL_RB: downlink Resource Block number in one slot
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_REs: all Cell Specific Reference Signal REs for 4 antenna, at least for the current symbol
    output:
        tuple of RE (k,l) that shall be used for PBCH
    Note:
        It is the caller's responsibility to ensure it is the correct slot.
        The REs are in the same sequence that they should be used in mapping PBCH symbols.
    '''
    result = tuple()
    
    for l in range(4):
        result += get_REs_for_PBCH_in_symbol(l, N_DL_RB, N_RB_sc, CSRS_REs)
    
    return result
#@-others


def get_PBCH_symbol_matrix(b, N_cell_ID, layer_mapping_scheme, num_of_layers, precoding_scheme, AP_num, codebook_index, N_DL_RB, N_RB_sc, CSRS_RE_tuple):
    '''
    input:
        b: channel coded PBCH bit sequence from higher layer.
        N_cell_ID: cell ID
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        AP_num: integer. The number of antenna ports configured for CSRS
        codebook_index: codebook index
        N_DL_RB: integer. Number of downlink RBs.
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_RE_tuple: a tuple of Cell Specific Reference Signal. Each element is represented as (k,l).
    output:
        matrix of symbols of all antenna ports for four OFDM symbols. e.g. output[0] is the symbol matrix for antenna port 0, and output[0][0] is the symbol array for ap=0 and l=0.
    '''
    scrambled_array = PBCH_scramble(b, N_cell_ID)
    modulated_array = PBCH_modulate(scrambled_array)
    layer_mapped_matrix = PBCH_layer_map(modulated_array, layer_mapping_scheme, num_of_layers)
    precoded_matrix = PBCH_precode(layer_mapped_matrix, precoding_scheme, AP_num, codebook_index)
    
    pbch_RE_tuple = get_REs_for_PBCH_in_slot(N_DL_RB, N_RB_sc, CSRS_RE_tuple)
    
    symbol_matrix = [0] * AP_num
    for ap in range(AP_num):
        re_array_for_4_symbols = array( [[0.0+0.0*1j] * (N_DL_RB*N_RB_sc)]*4 )
        precoded_symbol_array = precoded_matrix[ap]
        lte_assert(len(precoded_symbol_array)==4*len(pbch_RE_tuple), "Number of precoded symbol for AP=%s is %s, number of available PBCH REs is %s, and the former is not four times of the latter!"%(ap, len(precoded_symbol_array), len(pbch_RE_tuple)))
        count = 0
        for k,l in pbch_RE_tuple:
            re_array_for_4_symbols[l][k] = precoded_symbol_array[count]
            count += 1
        symbol_matrix[ap] = re_array_for_4_symbols


    return symbol_matrix
    

    
#@+node:Michael.20120319125504.1463: *5* 6.07 PCFICH
#@+others
#@+node:Michael.20120319125504.1464: *6* 6.7.1 Scrambing
def PCFICH_scramble(b, n_s, N_cell_ID):
    '''
    input:
        b: tuple of integer, the 32 bits. Note: len(b)==32 must be true
        n_s: slot index
        N_cell_ID: cell ID
    output: tuple of integer, the scrambled 32 bits
    '''
    c_init = (n_s/2+1) * (2*N_cell_ID+1) * (2**9) + N_cell_ID
    b_ = [0] * 32
    for i in range(32):
        b_[i] = (b[i] + c(c_init,i))%2
    return tuple(b_)
#@+node:Michael.20120319125504.1465: *6* 6.7.2 Modulation
def PCFICH_modulate(b_):
    '''
    input:
        b_: scrambled PCFICH sequence, 32 bits
    output:
        modulated 16 symbols (complex number)
    '''
    d = array([0.0 + 1j*0.0] * 16)
    for i in range(16):
        d[i] = QPSK( (b_[2*i], b_[2*i+1]) )
    return d
#@+node:Michael.20120319125504.1474: *6* 6.7.3 Layer mapping and precoding
def PCFICH_layer_map(d, layer_mapping_scheme, num_of_layers):
    '''
    input:
        b: array of 16 modulated symbols
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
    output:
        layer mapped array of arrays for each layer. output[0] is the array for layer 0, and so on.
    '''
    lte_assert(layer_mapping_scheme in ('single_antenna', 'transmit_diversity'), "layer_mapping_scheme=%s is not valid for PCFICH. It must be 'single_antenna' or 'transmit_diversity'."%(layer_mapping_scheme, ))
    if layer_mapping_scheme == 'single_antenna':
        lte_assert(num_of_layers==1, "For single antenna scheme of layer mapping, number of layers must be 1.")
        result = layer_map_single_antenna_port(d)
    else:
        lte_assert(num_of_layers in (2,4), "For transmit diversity scheme of layer mapping, number of layers must be either 2 or 4.")
        result = layer_map_transmit_diversity([d], num_of_layers)
    return result

def PCFICH_precode(layer_mapped_matrix, precoding_scheme, num_of_ap, codebook_index):
    '''
    input:
        layer_mapped_matrix: matrix after PCFICH layer mapping.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_ap: number of transmission antenna ports
        codebook_index: codebook index
    output:
        maxtrix for all transmission antenna ports. e.g. output[0] is the symbol array for ap_0.
    '''
    lte_assert(precoding_scheme in ('single_antenna', 'transmit_diversity'), "For PCFICH precoding, the scheme must either be 'single_antenna' or 'transmit_diversity', but the current scheme is %s"%precoding_scheme)
    num_of_layers = len(layer_mapped_matrix)
    if precoding_scheme == 'single_antenna':
        lte_assert(num_of_ap==1, "For PCFICH 'single_antenna' precoding scheme, num_of_ap must be 1, but currently it is %s."%num_of_ap)
        lte_assert(num_of_layers==1, "For PCFICH 'single_antenna' precoding scheme, number of layers must be 1, but currently it is %s."%num_of_layers)
        result = precode_single_antenna(layer_mapped_matrix)
    else:
        # precoding_scheme == 'transmit_diversity'
        lte_assert(num_of_ap in (2,4), "For PCFICH 'transmit_diversity' precoding scheme, number of transmission antenna must be 2 or 4. Current number of transmitting antenna is %s"%num_of_ap)
        lte_assert(num_of_ap==num_of_layers, "For transmit diversity precoding, number of transmitting antenna must be equal to number of layers, but currently num_of_ap=%s and num_of_layers=%s"%(num_of_ap, num_of_layers))
        result = precode_transmit_diversity(layer_mapped_matrix, num_of_ap)
    return result
#@+node:Michael.20120320091224.1503: *6* 6.7.4 Mapping to resource elements
def get_REG_for_PCFICH_quadruplet(quadruplet_index, N_cell_ID, N_DL_RB, N_RB_sc, CSRS_RE_tuple, num_of_ap, CP_DL_type):
    '''
    input:
        quadruplet_index: index of quadruplet (0..3)
        N_cell_ID: cell ID
        N_DL_RB: downlink Resource Block number in one slot
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_RE_tuple: a tuple of Cell Specific Reference Signal. Each element is represented as (k,l).
        num_of_ap: number of antenna ports used for Cell Specific Reference Signal
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        REG tuple for this PCFICH quadruplet: type(output) is tuple, len(output)==4
    '''
    #lte_assert(num_of_ap in (0,1,2,3), "Antenna port for PCFICH quadruplet must be 0, 1, 2, or 3, but it is assigned to %s"%ap)
    lte_assert(quadruplet_index in (0,1,2,3), "quadruplet_index of PCFICH must be 0, 1, 2, or 3, but it is %s"%quadruplet_index)
    
    k_ = (N_RB_sc/2) * (N_cell_ID%(2*N_DL_RB))
    k = int(k_ + floor(quadruplet_index * N_DL_RB/2.0)*N_RB_sc/2)
    l = 0   # it's always in the first OFDM symbol
    reg = get_REG_from_kl(num_of_ap, CP_DL_type, k, l)
    result = list()
    for re in reg:
        if re not in CSRS_RE_tuple:
            result.append(re)
    lte_assert(len(result)==4, "One REG must contain 4 available REs! The current REG is: %s"%result)
    return tuple(result)


        
#@-others

def valid_CFI_range(LTE_mode, is_MBSFN, support_PDSCH, AP_num, positioning_RS_enabled, N_DL_RB, subframe):
    '''
    valid_CFI_range(LTE_mode, is_MBSFN, support_PDSCH, positioning_RS_enabled, N_DL_RB): a tuple of valid CFI integers
        LTE_mode: 'TDD' or 'FDD'
        is_MBSFN: boolean. Is it a carrier supporting MBSFN?
        support_PDSCH: boolean. Does this carrier support PDSCH?
        AP_num: integer. The number of antenna ports configured for CSRS
        positioning_RS_enabled: boolean. Does positioning Reference Signal configured?
        N_DL_RB: integer. Number of downlink RBs.
        subframe: subframe index (0..9)
    This method is written according to table 6.7-1 of 36.211
    '''
    CFI_candidates = tuple()
    if LTE_mode=='TDD' and subframe in (1,6):
        if N_DL_RB>10:
            CFI_candidates = (1,2)
        elif N_DL_RB<=10:
            CFI_candidates = (2,)
    elif is_MBSFN and support_PDSCH and AP_num in (1,2):
        if N_DL_RB>10:
            CFI_candidates = (1,2)
        elif N_DL_RB<=10:
            CFI_candidates = (2,)
    elif is_MBSFN and support_PDSCH and AP_num==4:
        CFI_candidates = (2,)
    elif not support_PDSCH:
        CFI_candidates = (0,)
    elif (not is_MBSFN) and positioning_RS_enabled:
        if LTE_mode=='TDD' and subframe==6:
            if N_DL_RB>10:
                CFI_candidates = (1,2,3)
            elif N_DL_RB<=10:
                CFI_candidates = (2,3,4)
        else:
            if N_DL_RB>10:
                CFI_candidates = (1,2,3)
            elif N_DL_RB<=10:
                CFI_candidates = (2,3)
    else:
        if N_DL_RB>10:
            CFI_candidates = (1,2,3)
        elif N_DL_RB<=10:
            CFI_candidates = (2,3,4)

def get_PCFICH_symbol_array(cfi_b, n_s, N_cell_ID, layer_mapping_scheme, num_of_layers, precoding_scheme, AP_num, codebook_index, N_DL_RB, N_RB_sc, CSRS_RE_tuple, CP_DL_type):
    '''
    input:
        cfi_b: coded CFI value from higher layer. len(cfi_b)==32 must be true.
        n_s: slot index
        N_cell_ID: cell ID
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        AP_num: integer. The number of antenna ports configured for CSRS
        codebook_index: codebook index
        N_DL_RB: integer. Number of downlink RBs.
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_RE_tuple: a tuple of Cell Specific Reference Signal. Each element is represented as (k,l).
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        matrix of symbols of all antenna ports for one OFDM symbol. e.g. output[0] is the symbol array for antenna port 0, and output[0][0] is the symbol for subcarrier 0 in this l for antenna port 0.
    '''
    scrambled_array = PCFICH_scramble(cfi_b, n_s, N_cell_ID)
    modulated_array = PCFICH_modulate(scrambled_array)
    layer_mapped_matrix = PCFICH_layer_map(modulated_array, layer_mapping_scheme, num_of_layers)
    precoded_matrix = PCFICH_precode(layer_mapped_matrix, precoding_scheme, AP_num, codebook_index)
    
    symbol_array_for_all_ap = array( [[0.0+0.0*1j] * (N_DL_RB*N_RB_sc)]*AP_num )
    for quadruplet_index in range(4):
        reg = get_REG_for_PCFICH_quadruplet(quadruplet_index, N_cell_ID, N_DL_RB, N_RB_sc, CSRS_RE_tuple, AP_num, CP_DL_type)
        for i in range(4):
            k, l = reg[i]
            for ap in range(AP_num):
                symbol_array_for_all_ap[ap][k] = precoded_matrix[ap][quadruplet_index*4+i]
    return symbol_array_for_all_ap
    

    
    
#@+node:michael.20120305092148.1290: *5* 6.09 PHICH
def m_i( UL_DL_config, subframe ):
    m_i_table = (   (2,1,None,None,None,2,1,None,None,None),
                            (0,1,None,None,1,0,1,None,None,1),
                            (0,0,None,1,0,0,0,None,1,0),
                            (1,0,None,None,None,0,0,0,1,1,),
                            (0,0,None,None,0,0,0,0,1,1),
                            (0,0,None,0,0,0,0,0,1,0),
                            (1,1,None,None,None,1,1,None,None,1)
                    )
    return m_i_table[UL_DL_config][subframe]
#@+node:michael.20120305092148.1280: *5* 6.10 RS
#@+others
#@+node:michael.20120305092148.1281: *6* 6.10.1 CSRS
#@+others
#@+node:michael.20120305092148.1285: *7* 6.10.1.1 Seq gen
def r_l_ns(n_s, l, N_cell_ID, N_maxDL_RB, DL_CP_type):
    '''
    r_l_ns(l, n_s, N_cell_ID, N_maxDL_RB, DL_CP_type): list of complex symbols for CSRS signal in symbol index l of given slot.
    l: symbol index in given slot
    n_s: slot index
    N_cell_ID:  cell ID
    N_maxDL_RB: 110 for 20MHz
    DL_CP_type: CP type for downlink, 0 for normal CP and 1 for extended CP
    
    profile:
import cProfile
from math import sqrt
def main():
    for i in range(100):
        tmp = r_l_ns(i%20, i%7, i, 110, i%2)

cProfile.runctx('main()', globals(), locals())

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    44000    0.227    0.000  109.216    0.002 <ipython console>:1(c)
        1    0.001    0.001  109.585  109.585 <ipython console>:1(main)
      100    0.255    0.003  109.583    1.096 <ipython console>:1(r_l_ns)
    44000   46.434    0.001   46.434    0.001 <ipython console>:1(x_1)
    44000   62.555    0.001   62.555    0.001 <ipython console>:1(x_2)
    '''
    if DL_CP_type == 0: # normal DL CP
        N_CP = 1
    else:
        N_CP = 0
    c_init = 2**10 * (7*(n_s+1)+l+1) * (2*N_cell_ID+1) + 2*N_cell_ID + N_CP
    csrs_symbol_list = list()
    for m in range(2*N_maxDL_RB):
        real_part = 1/sqrt(2) * (1-2*c(c_init,2*m))
        image_part = 1/sqrt(2) * (1-2*c(c_init,2*m+1))
        csrs_symbol_list.append( complex(real_part,image_part) )
    return tuple(csrs_symbol_list)
#@+node:michael.20120305092148.1279: *7* 6.10.1.2 Mapping to REs
def get_CSRS_REs_in_slot(n_s, antenna_port, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb):
    '''
    get_CSRS_REs_in_slot(n_s, antenna_port, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb): tuple of CSRS REs in the specified symbol of RB.
    n_s: slot index
    antenna_port: antenna port for CSRS
    N_cell_ID: cell ID
    N_maxDL_RB: 110 for 20MHz configured by higher layer
    N_DL_RB: PHY number of downlink RB
    N_DL_symb: maximum 110 for 20MHz
    
    profile:
def main():
    for i in range(1000):
        tmp = get_CSRS_in_slot(i%20, i%4, i, 110, 110, (7,3,2,4,5,1)[i%6])

cProfile.runctx('main()', globals(), locals())

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.012    0.012    0.903    0.903 <ipython console>:1(main)
     1000    0.536    0.001    0.891    0.001 <ipython console>:2(get_CSRS_in_slot)
    '''
    
    REs = list()
    # symbol indices for CSRS of this AP
    if antenna_port in (0,1):
        if N_DL_symb>3:
            l_list = (0, N_DL_symb-3)
        else:   # DwPTS that has only 3 DL symbols
            l_list = (0,)
    else:   # antenna_port in (2,3)
        l_list = (1,)
    # v_shift
    v_shift = N_cell_ID % 6
    for l in l_list:
        # v
        if antenna_port==0 and l==0:
            v = 0
        elif antenna_port==0 and l!=0:
            v = 3
        elif antenna_port==1 and l==0:
            v = 3
        elif antenna_port==1 and l!=0:
            v = 0
        elif antenna_port==2:
            v = 3 * (n_s%2)
        elif antenna_port==3:
            v = 3 + 3 * (n_s%2)
        for m in range(2*N_DL_RB-1):
            m_ = m + N_maxDL_RB - N_DL_RB   # m'
            k = 6*m + (v+v_shift)%6
            REs.append( (k,l) )
    return tuple(REs)

def get_CSRS_in_symbol(n_s, l, antenna_port, N_cell_ID, N_maxDL_RB, N_DL_RB, N_RB_sc, N_DL_symb, DL_CP_type):
    '''
    '''
    symbol_array = ndarray(shape=(N_maxDL_RB*N_RB_sc,),dtype=complex128)
    for i in arange(len(symbol_array)):
        symbol_array[i] = 0
    csrs_seq = r_l_ns(n_s, l, N_cell_ID, N_maxDL_RB, DL_CP_type)
    # symbol indices for CSRS of this AP
    if antenna_port in (0,1):
        if N_DL_symb>3:
            l_list = (0, N_DL_symb-3)
        else:   # DwPTS that has only 3 DL symbols
            l_list = (0,)
    else:   # antenna_port in (2,3)
        l_list = (1,)
    # v_shift
    v_shift = N_cell_ID % 6
    if l in l_list:
        # v
        if antenna_port==0 and l==0:
            v = 0
        elif antenna_port==0 and l!=0:
            v = 3
        elif antenna_port==1 and l==0:
            v = 3
        elif antenna_port==1 and l!=0:
            v = 0
        elif antenna_port==2:
            v = 3 * (n_s%2)
        elif antenna_port==3:
            v = 3 + 3 * (n_s%2)
        for m in range(2*N_DL_RB):
            m_ = m + N_maxDL_RB - N_DL_RB   # m'
            k = 6*m + (v+v_shift)%6
            symbol_array[k] = csrs_seq[m_]
    return symbol_array
#@-others
#@-others
#@+node:michael.20120305092148.1301: *5* 6.11 Sync signals
#@+others
#@+node:michael.20120305092148.1302: *6* 6.11.1 PSS
#@+others
#@+node:michael.20120305092148.1303: *7* 6.11.1.1 seq gen
def pss_d(n, N_ID_2):
    u = (25, 29, 34)[N_ID_2]
    d_n = 0
    if n>=0 and n<=30:
        d_n = exp(-1j*pi*u*n*(n+1)/63)
    elif n>=31 and n<=61:
        d_n = exp(-1j*pi*u*(n+1)*(n+2)/63)
    return d_n
#@+node:michael.20120305092148.1305: *7* 6.11.1.2 mapping to REs
def pss_symbol_array(N_ID_2, N_DL_RB, N_RB_sc):
    symbol_array = ndarray( shape=(N_DL_RB*N_RB_sc,), dtype=complexfloating )
    for i in arange(len(symbol_array)):
        symbol_array[i] = 0
    for n in arange(0, 62):
        k = n-31+N_DL_RB*N_RB_sc/2
        symbol_array[k] = pss_d(n, N_ID_2)
    return symbol_array

def pss_k_range(N_DL_RB, N_RB_sc):
    
    start_index = 0-31+N_DL_RB*N_RB_sc/2
    end_index = 61-31+N_DL_RB*N_RB_sc/2
    
    return (start_index, end_index)

def get_pss_seq_from_RE_symbol_array(re_symbol_array, N_RB_sc):
    '''
    Note: len(re_symbol_array)==N_DL_RB*N_RB_sc must be True!
    '''
    pss_seq_received = array([0.0 + 0.0 * 1j] * (6 * N_RB_sc))
    pss_start_k, pss_end_k = pss_k_range(N_DL_RB, N_RB_sc)
    tmp_index = pss_start_k
    for i in arange(5, 6*N_RB_sc):
        pss_seq_received[i] = re_symbol_array[tmp_index]
        tmp_index += 1
    return pss_seq_received
#@-others
#@+node:michael.20120312091134.1403: *6* 6.11.2 SSS
#@+others
#@+node:michael.20120312091134.1404: *7* 6.11.2.1 Sequence generation
def sss_x5(mask, n):
    x_init = 0b10000
    while n>4:
        tmp = 0
        for i in (0,1,2,3,4):
            if (mask>>i)&1 == 1: # feedback for this item is enabled
                tmp += int((x_init>>i)&1)
            tmp = tmp%2
        x_init = x_init>>1 ^ tmp*(2**4)
        n -= 1
    return int((x_init>>n)&1)


def sss_z_(i):
    if i >=0 and i <=30:
        return 1 - 2 * sss_x5(0b10111, i)

def sss_z_1(m, n):
    return sss_z_((n+(m%8))%31)

def sss_c_(i):
    if i>=0 and i<=30:
        return 1 - 2 * sss_x5(0b01001, i)

def sss_c_0(n, N_ID_2):
    return sss_c_((n+N_ID_2)%31)

def sss_c_1(n, N_ID_2):
    return sss_c_((n+N_ID_2+3)%31)

def sss_s_(i):
    return 1 - 2 * sss_x5(0b00101, i)

def sss_s(m, n):
    return sss_s_((n+m)%31)

def sss_d(n, subframe, N_ID_cell):
    N_ID_1 = N_ID_cell/3
    N_ID_2 = N_ID_cell%3
    q_ = N_ID_1/30
    q = (N_ID_1 + q_*(q_+1)/2)/30
    m_ = N_ID_1 + q*(q+1)/2
    m_0 = m_%31
    m_1 = (m_0 + m_/31 + 1)%31
    if n%2==0:
        n_ = n/2
        if subframe == 0:
            result = sss_s(m_0, n_) * sss_c_0(n_, N_ID_2)
        elif subframe == 5:
            result = sss_s(m_1, n_) * sss_c_0(n_, N_ID_2)
    else:
        n_ = (n-1)/2
        if subframe == 0:
            result = sss_s(m_1, n_) * sss_c_1(n_, N_ID_2) * sss_z_1(m_0, n_)
        elif subframe == 5:
            result = sss_s(m_0, n_) * sss_c_1(n_, N_ID_2) * sss_z_1(m_1, n_)
    return result

def sss_seq(subframe, N_ID_cell):
    sss = array([0] * 62)
    for i in range(62):
        sss[i] = sss_d(i, subframe, N_ID_cell)
    return sss

#@+node:Michael.20120314113327.1416: *7* 6.11.2.2 Mapping to REs
def sss_symbol_array(subframe, N_ID_cell, N_DL_RB, N_RB_sc):
    symbol_array = ndarray( shape=(N_DL_RB*N_RB_sc,), dtype=complex128 )
    for i in arange(len(symbol_array)):
        symbol_array[i] = 0
    for n in arange(0, 62):
        k = n-31+N_DL_RB*N_RB_sc/2
        symbol_array[k] = sss_d(n, subframe, N_ID_cell)
    return symbol_array

def sss_k_range(N_DL_RB, N_RB_sc):
    
    start_index = 0-31+N_DL_RB*N_RB_sc/2
    end_index = 61-31+N_DL_RB*N_RB_sc/2
    
    return (start_index, end_index)

def get_sss_seq_from_RE_symbol_array(re_symbol_array, N_RB_sc, do_adjust=True, adjust_method='round_to_one'):
    '''
    Note: len(re_symbol_array)==N_DL_RB*N_RB_sc must be True!
    '''
    sss_start_k, sss_end_k = sss_k_range(N_DL_RB, N_RB_sc)
    tmp_index = sss_start_k
    if do_adjust:
        if adjust_method=='+1-1':
            sss_seq_received = array([0]*(6*N_RB_sc))
            for i in arange(5, 6*N_RB_sc-5):
                #if abs(real(re_symbol_array[tmp_index])) > abs(imag(re_symbol_array[tmp_index])):
                #print 're_symbol_array[%s]=%s'%(tmp_index, re_symbol_array[tmp_index])
                tmp_angle = angle(re_symbol_array[tmp_index])
                if tmp_angle>-0.5*pi and tmp_angle<=0.5*pi:
                    sss_seq_received[i] = 1
                else:
                    sss_seq_received[i] = -1
                tmp_index += 1
        elif adjust_method=='round_to_one':
            sss_seq_received = array([0.0 + 0.0*1j]*(6*N_RB_sc))
            for i in arange(5, 6*N_RB_sc-5):
                #if abs(real(re_symbol_array[tmp_index])) > abs(imag(re_symbol_array[tmp_index])):
                #print 're_symbol_array[%s]=%s'%(tmp_index, re_symbol_array[tmp_index])
                #tmp_angle = angle(re_symbol_array[tmp_index])
                sss_seq_received[i] = re_symbol_array[tmp_index]/abs(re_symbol_array[tmp_index])
                tmp_index += 1
    else:
        sss_seq_received = array([0.0 + 0.0 * 1j] * (6 * N_RB_sc))
        for i in arange(5, 6*N_RB_sc):
            sss_seq_received[i] = re_symbol_array[tmp_index]
            tmp_index += 1
    #print sss_seq_received
    return sss_seq_received
#@-others
#@-others
#@+node:michael.20120305092148.1293: *5* 6.12 OFDM baseband signal gen
def ofdm_baseband_IQ_signal_generate(symbol_array, l, N_DL_RB, N_RB_sc, N_DL_CP, delta_f=15000, gen_method='IFFT'):
    '''
    Note: len(symbol_array)==N_DL_RB*N_RB_sc must be True.
    '''
    T_s = 1./30720/1000  # all time scale is in 1 s
    if N_DL_CP==0 and delta_f==15000:
        if l==0:
            N_CP_l = 160
        else:
            N_CP_l = 144
    elif N_DL_CP==1:    # extended CP
        if delta_f==15000:
            N_CP_l = 512
        else:   # delta_f == 7500
            N_CP_l = 1024
    if delta_f==15000:
        N = 2048
    else:   # delta_f == 7500
        N = 4096
    t = arange(0, (N_CP_l+N)*T_s, T_s)
    signal_pl =  array([0.0+0.0*1j] * (N_CP_l + N))
    
    down_limit = int(floor(N_DL_RB*N_RB_sc/2))
    up_limit = int(ceil(N_DL_RB*N_RB_sc/2))
    
    if gen_method == 'DIRECT':
        for k in arange( -1*down_limit, 0, 1 ):
            signal_pl += symbol_array[k+down_limit]*exp(1j*2*pi*k*delta_f*(t-N_CP_l*T_s))
        for k in arange(1, up_limit+1, 1):
            signal_pl += symbol_array[k+down_limit-1]*exp(1j*2*pi*k*delta_f*(t-N_CP_l*T_s))
    elif gen_method == 'IFFT':
        mapped_seq = map_baseband_IQ_for_ifft(symbol_array)
        signal_pl[N_CP_l:] = fft.ifft(mapped_seq, N) * N
        signal_pl[:N_CP_l] = signal_pl[-1*N_CP_l:]
        
    return signal_pl

def map_baseband_IQ_for_ifft(baseband_IQ_array):
    '''
    Note: len(symbol_array)==N_DL_RB*N_RB_sc must be True.
    '''
    #T_s = 1./30720/1000  # all time scale is in 1 s
    if delta_f==15000:
        N = 2048
    else:   # delta_f == 7500
        N = 4096
    #t = arange(0, (N_CP_l+N)*T_s, T_s)
    #signal_pl =  array([0.0+0.0*1j] * (N_CP_l + N))
    
    down_limit = int(floor(len(baseband_IQ_array)/2))
    up_limit = int(ceil(len(baseband_IQ_array)/2))
    
    mapped_seq = array([0.0+0.0*1j] * N)
    # do the mapping before IFFT
    tmp_index = N-1
    for i in arange(down_limit-1, -1, -1):
        mapped_seq[tmp_index] = baseband_IQ_array[i]
        tmp_index -= 1
    tmp_index = 1
    for i in arange(down_limit, down_limit+up_limit):
        mapped_seq[tmp_index] = baseband_IQ_array[i]
        tmp_index += 1
    return mapped_seq

def map_fft_result_to_RE_IQ_array(fft_result_array):
    '''
    map_fft_result_to_baseband_IQ(fft_result_array): baseband_IQ_array
    Note: len(fft_result_array)==N must be True
            len(baseband_IQ_array) is N_DL_RB*N_RB_sc
    '''
    if delta_f==15000:
        N = 2048
    else:   # delta_f == 7500
        N = 4096
    
    mapped_seq = array([0.0+0.0*1j] * (N_DL_RB*N_RB_sc))
    
    down_limit = int(floor(len(mapped_seq)/2))
    up_limit = int(ceil(len(mapped_seq)/2))
    
    tmp_index = N-1
    for i in arange(down_limit-1, -1, -1):
        mapped_seq[i] = fft_result_array[tmp_index]
        tmp_index -= 1
    tmp_index = 1
    for i in arange(down_limit, down_limit+up_limit):
        mapped_seq[i] = fft_result_array[tmp_index]
        tmp_index += 1
        
    return mapped_seq
    
def ofdm_baseband_IQ_to_RE_IQ_array(baseband_IQ_array, N_DL_RB, N_RB_sc, delta_f=15000):
    '''
    Note: len(baseband_IQ_array)==N must be True.
    '''
    if delta_f==15000:
        N = 2048
    else:   # delta_f == 7500
        N = 4096

    re_IQ_array =  array([0.0+0.0*1j] * (N_DL_RB * N_RB_sc))
    re_IQ_array = 1.0/N * map_fft_result_to_RE_IQ_array(fft.fft(baseband_IQ_array, N))
        
    return re_IQ_array
#@+node:michael.20120305092148.1296: *5* 6.13 Modulation&upconversion
def downlink_modulate(s_p_l, t, f_0):
    modulated_signal = cos(2*pi*f_0*t) * s_p_l.real - sin(2*pi*f_0*t) * imag(s_p_l)
    cutoff_freq = f_0
    nyq = 2 * f_0
    numtaps = 80
    lp_fir = firwin(numtaps, cutoff_freq, window=('kaiser',8), nyq=nyq)
    filtered_modulated_signal = convolve( modulated_signal, lp_fir )[numtaps/2:len(modulated_signal)+numtaps/2]
    return modulated_signal

def downlink_downconvert(signal, t, f_0):
    
    cutoff_freq = f_0
    nyq = 2 * f_0
    numtaps = 80
    lp_fir = firwin(numtaps, cutoff_freq, window=('kaiser',8), nyq=nyq)
    
    I = -2* convolve( signal * cos(2*pi*f_0*t), lp_fir )[numtaps/2:len(signal)+numtaps/2]
    Q = -2 * convolve( signal * sin(2*pi*f_0*t), lp_fir )[numtaps/2:len(signal)+numtaps/2]
    
    return I + 1j*Q
#@-others
#@+node:michael.20120305092148.1294: *4* 7 Generic functions
#@+others
#@+node:Michael.20120319125504.1466: *5* 7.1 Modulation mapper
#@+others
#@+node:Michael.20120319125504.1467: *6* 7.1.1 BPSK
def BPSK(b):
    '''
    input:
        b: one bit, integer 0 or 1
    output:
        one complex symbol
    '''
    # one bit modulation
    return (1/sqrt(2) + 1j*1/sqrt(2), -1/sqrt(2) + 1j*(-1)/sqrt(2))[b]
#@+node:Michael.20120319125504.1468: *6* 7.1.2 QPSK
def QPSK((b0,b1)):
    '''
    input:
        (b0,b1): two element tuple, each represents one bit, must be either 0 or 1
    output:
        one complex modulated symbol
    '''
    return (complex(1/sqrt(2),1/sqrt(2)), complex(1/sqrt(2),-1/sqrt(2)),
                        complex(-1/sqrt(2),1/sqrt(2)), complex(-1/sqrt(2),-1/sqrt(2)))[2*b0+b1]


#@+node:Michael.20120319125504.1469: *6* 7.1.3 16QAM
def sixteenQAM((b0,b1,b2,b3)):
    '''
    input:
        (b0,b1,b2,b3): four element tuple, each represents one bit, must be 0 or 1
    output:
        one complex modulated symbol
    '''
    return (  complex(1*I, 1*Q),
                            complex(1*I, 3*Q),
                            complex(3*I, 1*Q),
                            complex(3*I, 3*Q),
                            complex(1*I, -1*Q),
                            complex(1*I, -3*Q),
                            complex(3*I, -1*Q),
                            complex(3*I, -3*Q),
                            complex(-1*I, 1*Q),
                            complex(-1*I,3*Q),
                            complex(-3*I, 1*Q),
                            complex(-3*I, 3*Q),
                            complex(-1*I, -1*Q),
                            complex(1*I, -3*Q),
                            complex(-3*I, -1*Q),
                            complex(-3*I, -3*Q),
    )[b3+2*b2+4*b1+8*b0]

#@+node:Michael.20120319125504.1470: *6* 7.1.4 64QAM
def sixtyFourQAM((b0,b1,b2,b3,b4,b5)):
    '''
    input:
        (b0,b1,b2,b3,b4,b5): six-element tuple, each of them represents one bit, must be 0 or 1
    output:
        one complex modulated symbol
    '''
    #tmp_64QAM = [0] * 64
    I = Q = 1/sqrt(42)
    x, y = (    (3,3), (3,1), (1,3), (1,1), (3,5), (3,7), (1,5), (1,7),
                        (5,3), (5,1), (7,3), (7,1), (5,5), (5,7), (7,5), (7,7),
                        (3,-3), (3,-1), (1,-3), (1,-1), (3,-5), (3,-7), (1,-5),(1,-7),
                        (5,-3), (5,-1), (7,-3), (7,-1), (5,-5), (5,-7), (7,-5), (7,-7),
                        (-3,3), (-3,1), (-1,3), (-1,1), (-3,5), (-3,7), (-1,5), (-1,7),
                        (-5,3), (-5,1), (-7,3), (-7,1), (-5,5), (-5,7), (-7,5), (-7,7),
                        (-3,-3), (-3,-1), (-1,-3),(-1,-1), (-3,-5), (-3,-7), (-1,-5), (-1,-7),
                        (-5,-3), (-5,-1), (-7,-3), (-7,-1), (-5,-5), (-5,-7), (-7,-5), (-7,-7) )[b5+b4*2+b3*4+b2*8+b1*16+b0*32]
    return x*I+1j*y*Q
#@-others
#@+node:michael.20120305092148.1283: *5* 7.2 Pseudo-random seq gen
def x_1(i):
    x1_init = 1
    while i>30:
        tmp = (x1_init&1 ^ x1_init&8)%2
        x1_init = x1_init>>1 ^ tmp*(2**30)
        i -= 1
    return (x1_init >> i) & 1

def x_2(c_init, i):
    while i>30:
        tmp = (c_init&1 ^ c_init&2 ^ c_init&4 ^ c_init&8)%2
        c_init = c_init>>1 ^ tmp*(2**30)
        i -= 1
    return (c_init >> i) & 1

def c(c_init, i):
    '''
    profile:
def main():
    for i in range(10000):
        c(i,i)

cProfile.runctx('main()', globals(), locals())

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    10000    0.059    0.000   91.819    0.009 <ipython console>:1(c)
        1    0.023    0.023   91.843   91.843 <ipython console>:1(main)
    10000   39.318    0.004   39.318    0.004 <ipython console>:1(x_1)
    10000   52.441    0.005   52.441    0.005 <ipython console>:1(x_2)
    '''
    N_C = 1600
    return (x_1(i+N_C) + x_2(c_init,i+N_C)) %2
#@-others
#@-others
#@+node:Michael.20120320091224.1509: *3* 36.212
#@+others
#@+node:Michael.20120321090100.1533: *4* 5.1.1 CRC calculation
def calc_CRC16(a):
    '''
    input:
        a: bit sequence
    output:
        bit sequence of length 16 for the CRC
    '''
    lte_warn("calc_CRC16 is only a dummy function!")
    return [0] * 16
#@+node:Michael.20120321090100.1534: *4* 5.1.3.1 Tail biting convolutional coding
def tail_biting_convolutional_code(c):
    '''
    input:
        c: bit sequence
    output:
        bit sequence of length 3*len(c)
    '''
    lte_warn("tail_biting_convolutional_code is only a dummy function!")
    
    result = array( [0] * (len(c)*3) )
    for i in range(len(c)):
        result[i] = c[i]
        result[2*i] = c[i]
        result[3*i] = c[i]
    return result
#@+node:Michael.20120321090100.1537: *4* 5.1.4.2 Rate matching for convolutionally coded TrCh and control information
def rate_match_for_conv_coded(d, E):
    '''
    input:
        d: bit sequence after convolutionary coding, which is serialized sequence for all three bit streams.
        E: target length after rate matchng.
    output:
        bit sequence of length E
    '''
    lte_warn("rate_match_for_conv_coded is only a dummy function!")
    
    e = array( [0] * E )
    D = len(d)
    for i in range(E):
        e[i] = d[i%D]
    
    return e
#@+node:Michael.20120321090100.1530: *4* 5.3 DL transport channels and control information
#@+others
#@+node:Michael.20120321090100.1531: *5* 5.3.1 Broadcast channel
#@+others
#@+node:Michael.20120321090100.1532: *6* 5.3.1.1 TB CRC attachment
def attach_BCH_TB_CRC(a, AP_num):
    '''
    input:
        a: array of bits for one BCH transport block.
        AP_num: number of transmitting antenna ports used for PBCH
    output:
        bit array with CRC attached.
    '''
    A = 24
    L = 16
    lte_assert(len(a)==A, "One BCH transport block must be %s bits long, but we've got %s bits instead."%(A, len(a)))
    
    c = array( [0]*(A+L) )
    crc16 = array( calc_CRC16(a) )
    lte_assert(len(crc16)==16, "16 bit CRC is actually %s bits!"%len(crc16))
    
    pbch_CRC_mask = array( (
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
                                    (0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1),
        )[AP_num>>1] )
    crc16 = crc16 ^ pbch_CRC_mask
    
    for i in range(len(crc16)):
        c[-1*i-1] = crc16[-1*i-1]
    
    return c
#@+node:Michael.20120321090100.1535: *6* 5.3.1.2 Channel coding
def PBCH_channel_code(c):
    '''
    input:
        c: bit sequence after PBCH CRC attachment.
    output:
        bit sequence
    '''
    A= 24
    L = 16
    K = A + L
    lte_assert(len(c)==K, "PBCH TB size before channel coding shall be %s, but it is %s"%(K, len(c)))
    return tail_biting_convolutional_code(c)
#@+node:Michael.20120321090100.1536: *6* 5.3.1.3 Rate matching
def PBCH_rate_match(d, CP_DL_type):
    '''
    input:
        d: bit sequence after PBCH channel coding.
        CP_DL_type: downlink CP type. 0 for normal CP
    output:
        bit sequence
    '''
    lte_assert(CP_DL_type in (0,1), "CP_DL_type has to be 0 for normal CP or 1 for extended CP, but it is %s."%CP_DL_type)
    
    if CP_DL_type == 0:
        # normal CP in downlink
        E = 1920
    else:
        # extended CP in downlink
        E = 1728
    
    result = rate_match_for_conv_coded(d, E)
    return result
#@-others

def BCH_channel_process(a, AP_num, CP_DL_type):
    '''
    input:
        a: array of bits for one BCH transport block. len(a) must be 24 bits.
        AP_num: number of transmitting antenna ports used for PBCH
        CP_DL_type: downlink CP type. 0 for normal CP
    output:
        bit sequence of length 1920 for normal DL CP, 1728 for extended DL CP.
    '''
    c = attach_BCH_TB_CRC(a, AP_num)
    d = PBCH_channel_code(c)
    e = PBCH_rate_match(d, CP_DL_type)
    return e
#@+node:Michael.20120320091224.1510: *5* 5.3.4 Control format indicator
def channel_code_CFI(cfi):
    '''
    input:
        cfi: cfi value from higher layer
    output:
        array of encoded cfi value. len(output)==32.
    '''
    lte_assert(cfi in (1,2,3), "CFI has to be 1, 2, or 3. Value 4 is reserved, according to table 5.3.4-1 in 36.212. Current CFI is %s"%cfi)
    encoded_cfi = (
                            (0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1),
                            (1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0),
                            (1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1)
                    )[cfi]
    return encoded_cfi
#@-others
#@-others
#@+node:Michael.20120319125504.1481: *3* Error handling
class LteException(Exception):
    
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return repr(self.value)
    
def lte_assert(condition, a_string):
    if not condition:
        raise LteException(a_string)

def lte_warn(a_string):
    print a_string
#@-others

test_enabling_bits = 0b11

# 01. PBCH symbol array in one OFDM symbol
if test_enabling_bits & (1<<0):
    test_PBCH_symbol_array_in_one_OFDM_symbol()

# 02. PBCH Uu signal
if test_enabling_bits & (1<<1):
    test_PBCH_Uu_signal()
#@-others
#@-leo
