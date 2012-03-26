#@+leo-ver=5-thin
#@+node:Michael.20120321090100.1528: * @thin ./Simulation/PBCH/pbch_gen_detect.py
#@+others
#@+node:Michael.20120321090100.1527: ** PBCH source
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
    input:
        n_PRB: PRB index
        N_RB_sc: number of subcarriers in one RB
        l: symbol index
        CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
        DL_CP_type: 0 for normal CP, 1 for extended CP
        optimize: boolean. whether to use optimized code for speed.
    output:
        a tuple of all REGs in the given PRB and symbol
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
#@+node:michael.20120323224953.1793: *6* get_RE_number_in_REG
def get_RE_number_in_REG(l, CSRS_AP_num, CP_DL_type):
    '''
    input:
        l: symbol index
        CSRS_AP_num: number of antenna ports used for CSRS.
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        number of REs that one REG contains, including those REs used for CSRS.
    '''
    result = 0
    if l == 0:
        result = 6
    elif l == 1:
        if CSRS_AP_num == 4:
            result = 6
        else:
            # CSRS_AP_num in (1,2)
            result = 4
    elif l == 2:
        result = 4
    elif l == 3:
        if CP_DL_type == 0:    # for nomal CP
            result = 4
        else:
            result = 6
    return result
#@+node:Michael.20120320091224.1504: *6* get_REG_from_kl
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
#@+node:michael.20120323224953.1794: *6* get_REG_number_in_symbol
def get_REG_number_in_symbol(N_DL_RB, N_RB_sc, l, CSRS_AP_num, DL_CP_type):
    '''
    input:
        N_DL_RB: number of downlinke resource blocks
        N_RB_sc: number of subcarriers in one RB
        l: symbol index
        CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        number of total REG number in the given symbol across all RBs.
    '''
    return N_DL_RB * N_RB_sc / get_RE_number_in_REG(l, CSRS_AP_num, CP_DL_type)
#@+node:michael.20120323224953.1795: *6* get_REG_in_symbol
def get_REG_in_symbol(N_DL_RB, N_RB_sc, l, CSRS_AP_num, DL_CP_type, CSRS_REs):
    '''
    input:
        N_DL_RB: downlink RB number
        N_RB_sc: number of subcarriers in one RB
        l: symbol index
        CSRS_AP_num: number of antenna uesed for Cell Specific Reference Signal
        DL_CP_type: 0 for normal CP, 1 for extended CP
        optimize: boolean. whether to use optimized code for speed.
        CSRS_REs: all REs used for CSRS in this symbol.
    output:
        a tuple of all REGs in the given symbol, from lowest frequency to the highest.
    '''
    result = list()
    for n_PRB in range(N_DL_RB):
        for reg in get_REG_in_RB_symbol(n_PRB, N_RB_sc, l, CSRS_AP_num, DL_CP_type):
            if len(reg) == 4:
                # there's no CSRS REs in here
                result.append(reg)
            else:
                # get rid of those two CSRS REs
                tmp_reg = list()
                for re in reg:
                    if re not in CSRS_REs:
                        tmp_reg.append(re)
                result.append( tuple(tmp_reg) )
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
    lte_assert( num_of_cw==1, "num_of_cw=%s is not correct for transmit diversity layer mapping! It must be 1!"%num_of_cw )
    lte_assert( v in (2,4), "v=%s is out of range! For transmit diversity layer mapping, number of layers has to be either 2 or 4!"%v)
        
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
            y[0][2*i] = 1/sqrt(2) * sum(array([1,0,1j,0]) * tmp_x)
            y[1][2*i] = 1/sqrt(2) * sum(array([0,-1,0,1j]) * tmp_x)
            y[0][2*i+1] = 1/sqrt(2) * sum(array([0,1,0,1j]) * tmp_x)
            y[1][2*i+1] = 1/sqrt(2) * sum(array([1,0,-1j,0]) * tmp_x)
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
            y[0][4*i] = 1/sqrt(2) * sum(array([1,0,0,0,1j,0,0,0]) * tmp_x)
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


        
# helper functions not in standard
def get_PCFICH_REG_num_in_symbol(l, is_PDCCH_present):
    '''
    input:
        l: symbol index
        is_PDCCH_present: boolean, whether PDCCH is present in the subframe containing symbol l
    output:
        number of REGs that are occupied by PCFICH in the given symbol
    '''
    if is_PDCCH_present and l==0:
        result = 3
    else:
        result = 0
    return result

def get_PCFICH_REs_in_symbol(l, is_PDCCH_present, N_cell_ID, N_DL_RB, N_RB_sc, CSRS_RE_tuple, num_of_ap, CP_DL_type):
    '''
    input:
        l: symbol index
        is_PDCCH_present: whether PDCCH is present in this slot
        N_cell_ID: cell ID
        N_DL_RB: downlink Resource Block number in one slot
        N_RB_sc: number of subcarriers in one Resource Block
        CSRS_RE_tuple: a tuple of Cell Specific Reference Signal. Each element is represented as (k,l).
        num_of_ap: number of antenna ports used for Cell Specific Reference Signal
        CP_DL_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        a tuple of all REs used by PCFICH in the given symbol
    '''
    result = list()
    if l==0 and is_PDCCH_present:
        for quadruplet_index in range(4):
            for re in get_REG_for_PCFICH_quadruplet(quadruplet_index, N_cell_ID, N_DL_RB, N_RB_sc, CSRS_RE_tuple, AP_num, CP_DL_type):
                result.append(re)
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
    return CFI_candidates

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


    
    
#@+node:Michael.20120326140330.1400: *5* 6.08 PDCCH
'''
table 6.8.1-1: supported PDCCH formats.

PDCCH format    Num. of CCEs    Num. of REGs    Num. of PDCCH bits
0                       1                       9                   72
1                       2                       18                  144
2                       4                       36                  288
3                       8                       72                  576
'''

class PDCCH:
    '''
    Represents one PDCCH channel by giving information of its: bit sequence to be transmitted, format shall be used, start_CCE_index of this PDCCH channel
    '''
    def __init__(self, bit_sequence, PDCCH_format, start_CCE_index):
        '''
        input:
            bit_sequence: a list of bits that's to be transmitted
            PDCCH_format: format of PDCCH (0..3)
            start_CCE_index: the starting CCE index of this PDCCH
        '''
        lte_assert(PDCCH_format in (0,1,2,3), "PDCCH format=%s is invalid! It must be 0, 1, 2, or 3, according to table 6.8.1-1 in 36.211."%PDCCH_format)
        if PDCCH_format == 0:
            lte_assert(len(bit_sequence)<=72, "PDCCH format 0 can only be used to transmit <=72 bits, but len(bit_sequence)=%s"%len(bit_sequence))
        elif PDCCH_format == 1:
            lte_assert(len(bit_sequence)<=144, "PDCCH format 1 can only be used to transmit <=144 bits, but len(bit_sequence)=%s"%len(bit_sequence))
            lte_assert(start_CCE_index%2==0, "PDCCH format 1 uses 2 CCEs, thus start_CCE_index%2 must be 0, but start_CCE_index=%s"%start_CCE_index)
        elif PDCCH_format == 2:
            lte_assert(len(bit_sequence)<=288, "PDCCH format 1 can only be used to transmit <=288 bits, but len(bit_sequence)=%s"%len(bit_sequence))
            lte_assert(start_CCE_index%4==0, "PDCCH format 1 uses 4 CCEs, thus start_CCE_index%4 must be 0, but start_CCE_index=%s"%start_CCE_index)
        else:
            # PDCCH_format == 3
            lte_assert(len(bit_sequence)<=576, "PDCCH format 1 can only be used to transmit <=576 bits, but len(bit_sequence)=%s"%len(bit_sequence))
            lte_assert(start_CCE_index%8==0, "PDCCH format 1 uses 8 CCEs, thus start_CCE_index%8 must be 0, but start_CCE_index=%s"%start_CCE_index)
        
        self.bit_sequence = bit_sequence
        self.format = PDCCH_format
        self.start_CCE_index = start_CCE_index

#@+others
#@+node:Michael.20120326140330.1401: *6* 6.8.2 PDCCH multiplexing and scrambing
def PDCCH_multiplex_and_scramble(pdcch_list, N_REG, n_s, N_cell_ID):
    '''
    input:
        pdcch_list: a list of instances of class PDCCH, which are to be transmitted on PDCCH in the given subframe. pdcch_list[0] shall be a instance of class PDCCH, thus pdcch_list[0].bit_sequence is a list of bits (e.g. (0,1,1,0,0,1,...)) that to be transmitted on PDCCH 0; pdcch_list[0].format is the format of this PDCCH, which shall be 0, 1, 2, or 3; pdcch_list[0].start_CCE_index is the starting CCE index of this PDCCH that must satisfies start_CCE_index%(2**format)==0.
        N_REG: total number of available REGs for PDCCH in the given subframe.
        n_s: slot index
        N_cell_ID: cell ID
    output:
        bit sequence of PDCCH in this subframe, its length is 8 * N_REG
    '''
    # the number of PDCCHs transmitted in the given subframe
    n_PDCCH = len(pdcch_list)
    
    # M_bit[i] is the lengt of how many bits will be transmitted on PDCCH i
    M_bit = [0] * n_PDCCH
    for i in range(n_PDCCH):
        M_bit[i] = len(pdcch_list[i].bit_sequence)
    
    # padding
    M_tot = 8 * N_REG
    lte_assert(M_tot>=sum(M_bit), "The total number of bits to be transmitted on PDCCH is larger than its capacity! N_REG=%s, PDCCH_capacity=8*N_REG=%s, total_bit_to_be_transmitted=%s"%(N_REG, 8*N_REG, sum(M_bit)))
    NIL = 0
    bit_sequence = [NIL] * M_tot
    for i in range(n_PDCCH):
        CCE_index = pdcch_list[i].start_CCE_index
        for j in range(len(pdcch_list[i].bit_sequence)):
            bit_sequence[72*CCE_index+j] = pdcch_list[i].bit_sequence[j]
    
    # scrambling
    c_init = n_s/2 * (2**9) + N_cell_ID
    for i in range(M_tot):
        bit_sequence[i] = (bit_sequence[i] + c(c_init, i)) % 2
    
    return bit_sequence
#@+node:Michael.20120326140330.1402: *6* 6.8.3 Modulation
def PDCCH_modulate(bit_sequence):
    '''
    input:
        bit_sequence: bit sequence of PDCCH in this subframe
    output:
        symbol sequence, e.g. output[0] is the 1st modulated symbol
    '''
    modulated_symbol_array = array( [0.0+0.0*1j] * (len(bit_sequence)/2) )
    for i in range(len(modulated_symbol_array)):
        modulated_symbol_array[i] = QPSK( (bit_sequence[2*i], bit_sequence[2*i+1]) )
    return modulated_symbol_array
#@+node:Michael.20120326140330.1403: *6* 6.8.4 Layer mapping and precoding
def PDCCH_layer_map(d, layer_mapping_scheme, num_of_layers):
    '''
    input:
        d: array of modulated symbols
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
    output:
        layer mapped array of arrays for each layer. output[0] is the array for layer 0, and so on.
    '''
    lte_assert(layer_mapping_scheme in ('single_antenna', 'transmit_diversity'), "layer_mapping_scheme=%s is not valid for PDCCH. It must be 'single_antenna' or 'transmit_diversity'."%(layer_mapping_scheme, ))
    if layer_mapping_scheme == 'single_antenna':
        lte_assert(num_of_layers==1, "For single antenna scheme of layer mapping, number of layers must be 1.")
        result = layer_map_single_antenna_port(d)
    else:
        lte_assert(num_of_layers in (2,4), "For transmit diversity scheme of layer mapping, number of layers must be either 2 or 4.")
        result = layer_map_transmit_diversity([d], num_of_layers)
    return result

def PDCCH_precode(layer_mapped_matrix, precoding_scheme, num_of_ap, codebook_index):
    '''
    input:
        layer_mapped_matrix: matrix after PDCCH layer mapping.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_ap: number of transmission antenna ports
        codebook_index: codebook index
    output:
        maxtrix for all transmission antenna ports. e.g. output[0] is the symbol array for ap_0.
    '''
    lte_assert(precoding_scheme in ('single_antenna', 'transmit_diversity'), "For PDCCH precoding, the scheme must either be 'single_antenna' or 'transmit_diversity', but the current scheme is %s"%precoding_scheme)
    num_of_layers = len(layer_mapped_matrix)
    if precoding_scheme == 'single_antenna':
        lte_assert(num_of_ap==1, "For PDCCH 'single_antenna' precoding scheme, num_of_ap must be 1, but currently it is %s."%num_of_ap)
        lte_assert(num_of_layers==1, "For PDCCH 'single_antenna' precoding scheme, number of layers must be 1, but currently it is %s."%num_of_layers)
        result = precode_single_antenna(layer_mapped_matrix)
    else:
        # precoding_scheme == 'transmit_diversity'
        lte_assert(num_of_ap in (2,4), "For PDCCH 'transmit_diversity' precoding scheme, number of transmission antenna must be 2 or 4. Current number of transmitting antenna is %s"%num_of_ap)
        lte_assert(num_of_ap==num_of_layers, "For transmit diversity precoding, number of transmitting antenna must be equal to number of layers, but currently num_of_ap=%s and num_of_layers=%s"%(num_of_ap, num_of_layers))
        result = precode_transmit_diversity(layer_mapped_matrix, num_of_ap)
    return result
#@+node:Michael.20120326140330.1404: *6* 6.8.5 Mapping to resource elements
def PDCCH_symbol_quadruplet_permute(precoded_matrix):
    '''
    input:
        precoded_matrix: symbol matrix after precoding, e.g. precoded_matrix[0] is the symbol sequence for antenna port 0.
    output:
        symbol quadruplet matrix after the permutation, e.g. output[0] is the symbol-quadruplet sequence for antenna port 0 after symbol quadruplet permutation; output[0][0] is the first symbol quadruplet for antenna port 0.
    '''
    lte_warn("PDCCH_symbol_quadruplet_permute is only a dummy function!")
    symbol_quadruplet_matrix = [0] * len(precoded_matrix)
    for i in range(len(precoded_matrix)):
        tmp_quadruplet_list = list()
        for j in range(len(precoded_matrix[i])/4):
            tmp_quadruplet_list.append( [ precoded_matrix[i][4*j], precoded_matrix[i][4*j+1], precoded_matrix[i][4*j+2], precoded_matrix[i][4*j+3] ] )
        symbol_quadruplet_matrix[i] = tmp_quadruplet_list
    return symbol_quadruplet_matrix

def PDCCH_symbol_quadruplet_cyclic_shift(symbol_quadruplet_matrix, N_cell_ID):
    '''
    input:
        symbol_quadruplet_matrix: symbol quadruplet matrix after symbol quadruplet permutation
        N_cell_ID: cell ID
    output:
        symbol quadruplet matrix after cyclic shift, e.g. output[0] is the symbol quadruplet array for antenna port 0; output[0][0] is the first symbol quadruplet of antenna port 0.
    '''
    M_quad = len(symbol_quadruplet_matrix[0])
    result = [0] * len(symbol_quadruplet_matrix)
    for i in range(len(symbol_quadruplet_matrix)):
        tmp_symbol_quadruplet_list = [0.0+0.0*1j] * M_quad
        for j in range(M_quad):
            tmp_symbol_quadruplet_list[j] = symbol_quadruplet_matrix[i][(j+N_cell_ID)%M_quad]
        result[i] = tmp_symbol_quadruplet_list
    return result

def get_PDCCH_RE_sequence(CFI, N_DL_RB, N_RB_sc, CSRS_REs, PCFICH_REs, PHICH_REs):
    '''
    input:
        CFI: Control  Format Indicator for this subframe.
        N_DL_RB: downlink RB number
        N_RB_sc: number of subcarriers in one RB
        CSRS_REs: a list of REs for Cell Specific Reference Signals in al relative symbols
        PCFICH_REs: a list of REs for PCFICH in this subframe
        PHICH_REs: a list of REs for PHICH in this subframe
    output:
        a array of PDCCH REs in this subframe, which is in the same sequence as PDCCH symbols shall be mapped.
    '''
    lte_assert(CFI in (1,2,3,4), "CFI=%s is out of range!"%CFI)
    L = CFI
    occupied_REs = CSRS_REs + PCFICH_REs + PHICH_REs
    result = list()
    m, k, l = 0, 0, 0
    while k < N_DL_RB * N_RB_sc:
        if ((k,l) not in occupied_REs) and ((k,l) not in result):
            result.append( (k,l) )
            re_count = 1    # number of valid REs found in this valid REG
            i = 1
            while re_count<4 and i<6:
                if ((k+i,l) not in occupied_REs) and ((k+i,l) not in result):
                    result.append( (k+i,l) )
                    re_count += 1
                    i += 1
                else:
                    i += 1
        else:
            if l < L-1:
                l += 1
            else:
                k += 1
                l = 0
    #print "PDCCH_REs = ", result
    return result
  
#@-others

def get_PDCCH_symbol_array(pdcch_list, CFI, N_REG, n_s, N_cell_ID, layer_mapping_scheme, num_of_layers, precoding_scheme, AP_num, codebook_index, N_DL_RB, N_RB_sc, CSRS_REs, PCFICH_REs, PHICH_REs):
    '''
    input:
        pdcch_list: a list of instances of class PDCCH, which are to be transmitted on PDCCH in the given subframe. pdcch_list[0] shall be a instance of class PDCCH, thus pdcch_list[0].bit_sequence is a list of bits (e.g. (0,1,1,0,0,1,...)) that to be transmitted on PDCCH 0; pdcch_list[0].format is the format of this PDCCH, which shall be 0, 1, 2, or 3; pdcch_list[0].start_CCE_index is the starting CCE index of this PDCCH that must satisfies start_CCE_index%(2**format)==0.
        CFI: Control  Format Indicator for this subframe.
        N_REG: total number of available REGs for PDCCH in the given subframe.
        n_s: slot index
        N_cell_ID: cell ID
        layer_mapping_scheme: 'single_antenna' or 'transmit_diversity'
        num_of_layers: number of layers.
        precoding_scheme: 'single_antenna' or 'transmit_diversity'
        AP_num: number of transmission antenna ports
        codebook_index: codebook index
        N_DL_RB: downlink RB number
        N_RB_sc: number of subcarriers in one RB
        CSRS_REs: a list of REs for Cell Specific Reference Signals in al relative symbols
        PCFICH_REs: a list of REs for PCFICH in this subframe
        PHICH_REs: a list of REs for PHICH in this subframe
    output:
        output[0] is a RE matrix for antenna port 0
        output[0][0] is a RE array of all subcarriers in symbol 0 for antenna port 0
    '''
    pdcch_bit_seq = PDCCH_multiplex_and_scramble(pdcch_list, N_REG, n_s, N_cell_ID)
    pdcch_symbol_seq = PDCCH_modulate(pdcch_bit_seq)
    pdcch_layer_mapped_matrix = PDCCH_layer_map(pdcch_symbol_seq, layer_mapping_scheme, num_of_layers)
    pdcch_precoded_matrix = PDCCH_precode(pdcch_layer_mapped_matrix, precoding_scheme, AP_num, codebook_index)
    permuted_symbol_quadruplet_matrix = PDCCH_symbol_quadruplet_permute(pdcch_precoded_matrix)
    cyclic_shifted_symbol_quadruplet_matrix = PDCCH_symbol_quadruplet_cyclic_shift(permuted_symbol_quadruplet_matrix, N_cell_ID)
    
    pdcch_RE_list = get_PDCCH_RE_sequence(CFI, N_DL_RB, N_RB_sc, CSRS_REs, PCFICH_REs, PHICH_REs)
    
    lte_assert(AP_num==len(cyclic_shifted_symbol_quadruplet_matrix), "AP_num!=len(cyclic_shifted_symbol_quadruplet_matrix), i.e. the number of transmitting antenna is not equal to antenna ports configuration in precoding!.")
    for i in range(AP_num):
        lte_assert(len(pdcch_RE_list)==4*len(cyclic_shifted_symbol_quadruplet_matrix[i]), "For antenna port %s, its symbol count (%s) doesn't match with PDCCH RE count (%s)!"%(i, 4*len(cyclic_shifted_symbol_quadruplet_matrix[i]), len(pdcch_RE_list)))
        
    symbol_array_for_all_ap = array( [ [ [0.0+0.0*1j] * (N_DL_RB*N_RB_sc) ] * CFI ] *AP_num )
    for ap in range(AP_num):
        for symbol_quadruplet_index in range(len(cyclic_shifted_symbol_quadruplet_matrix[ap])):
            for i in range(4):  # for the 4 symbols in this symbol quadruplet
                k, l = pdcch_RE_list[4*symbol_quadruplet_index+i]
                symbol_array_for_all_ap[ap][l][k] = cyclic_shifted_symbol_quadruplet_matrix[ap][symbol_quadruplet_index][i]
    
    return symbol_array_for_all_ap

#@+node:michael.20120305092148.1290: *5* 6.09 PHICH
def get_m_i( UL_DL_config, subframe ):
    m_i_table = (   (2,1,None,None,None,2,1,None,None,None),
                            (0,1,None,None,1,0,1,None,None,1),
                            (0,0,None,1,0,0,0,None,1,0),
                            (1,0,None,None,None,0,0,0,1,1,),
                            (0,0,None,None,0,0,0,0,1,1),
                            (0,0,None,0,0,0,0,0,1,0),
                            (1,1,None,None,None,1,1,None,None,1)
                    )
    return m_i_table[UL_DL_config][subframe]
    
    
def get_PHICH_symbol_array(HI, subframe, n_s, n_seq_PHICH, n_group_PHICH, N_group_PHICH, PHICH_duration, N_cell_ID, DL_CP_type, AP_num, LTE_mode, UL_DL_config, N_maxDL_RB, N_DL_RB, N_RB_sc, CSRS_AP_num, is_MBSFN, N_DL_symb):
    '''
    input:
        HI: HARQ Indicator, 0 for NACK, 1 for ACK
        subframe: subframe index
        n_s: slot index
        n_seq_PHICH: PHICH orthogonal sequence number within the PHICH group
        n_group_PHICH: PHICH group index
        N_group_PHICH: total PHICH group number, from higher layer
        PHICH_duration: PHICH duration from higher layer, 0 for normal duration, 1 for extended duration
        N_cell_ID: cell ID
        DL_CP_type: downlink CP type, 0 for normal CP, 1 for extended CP
        AP_num: number of antenna ports used to transmit PHICH
        LTE_mode: 'FDD' or 'TDD'
        UL_DL_config: Uplink/Downlink configuration, only valid if LTE_mode is 'TDD'
        N_maxDL_RB: Maximum DL RB number
        N_DL_RB: DL RB number
        N_RB_sc: number of subcarriers in one RB
        CSRS_AP_num: number of antenna ports used for CSRS
        is_MBSFN: boolean, whether it is a subframe for MBSFN
        N_DL_symb: number of symbols in one slot
    output:
        output[0] is a RE matrix for antenna port 0
        output[0][0] is a RE array of all subcarriers in symbol 0 for antenna port 0
    '''
    # calc m_i
    if LTE_mode == 'FDD':
        m_i = 0
    else:
        m_i = get_m_i( UL_DL_config, subframe )
    # channel coding
    b = channel_code_HI(HI)
    # modulation
    d = PHICH_modulate(b, n_s, n_seq_PHICH, N_cell_ID, DL_CP_type)
    # resource group alignment, layer mapping, and precoding
    d_0 = PHICH_align(d, n_group_PHICH, DL_CP_type, AP_num)
    layer_mapped_matrix = PHICH_layer_map_and_precode(d_0, AP_num, n_group_PHICH, DL_CP_type)
    
    mapping_units = get_PHICH_mapping_units(LTE_mode, N_maxDL_RB, N_DL_RB, N_RB_sc, CSRS_AP_num, m_i, N_group_PHICH, DL_CP_type, is_MBSFN, PHICH_duration, subframe, n_s, N_cell_ID, N_DL_symb)
    
    symbol_matrix_for_all_ap = array( [ [[0.0+0.0*1j] * (N_DL_RB*N_RB_sc)]*3 ]*AP_num )
    # symbol_array_for_all_ap[0] is the RE matrix for ap 0
    # symbol_array_for_all_ap[0][0] is the RE array in symbol 0 for ap 0
    
    m = n_group_PHICH
    if DL_CP_type == 0:
        m_ = m
    else:
        m_ = m/2
    reg_list = mapping_units[m_]
    for reg_index in range(3):  # it has to be 3 REG
        for re_index in range(4):   # it has to be 4 REs in one REG
            k, l = reg_list[reg_index][re_index]
            for ap in range(AP_num):
                symbol_matrix_for_all_ap[ap][l][k] = layer_mapped_matrix[ap][reg_index*4+re_index]

    return symbol_matrix_for_all_ap
#@+node:Michael.20120323090727.1968: *6* 6.9.1 Modulation
def PHICH_modulate(b, n_s, n_seq_PHICH, N_cell_ID, DL_CP_type):
    '''
    input:
        b: bit sequence
        n_s: slot index
        n_seq_PHICH: PHICH number within the PHICH group
        N_cell_ID: cell ID
        DL_CP_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        modulated PHICH multipled with a orthogonal sequence
    '''
    lte_assert(DL_CP_type in (0,1), "DL CP type DL_CP_type=%s is out of range. It has to be either 0 (normal CP) or 1 (extended CP)"%DL_CP_type)
    
    M_bit = len(b)
    
    # BPSK modulation
    M_s = M_bit
    z = array( [0.0+0.0*1j] * M_s )
    for i in range(M_bit):
        z[i] = BPSK(b[i])
    
    # multiply with the orthogonal sequence
    if DL_CP_type == 0:
        N_PHICH_SF = 4
    else:
        # extended CP
        N_PHICH_SF = 2
    M_symb = N_PHICH_SF * M_s
    c_init = (n_s/2+1) * (2*N_cell_ID+1) * (2**9) + N_cell_ID
    w = get_orthogonal_seq_for_PHICH(n_seq_PHICH, N_PHICH_SF)
    d = array( [0.0+0.0*1j] * M_symb )
    for i in range(M_symb):
        d[i] = w[i%N_PHICH_SF] * (1-2*c(c_init,i)) * z[i/N_PHICH_SF]
    
    return d


def get_orthogonal_seq_for_PHICH(n_seq_PHICH, N_PHICH_SF):
    '''
    input:
        n_seq_PHICH: sequence index, corresponding to the PHICH number within the PHICH group.
        N_PHICH_SF: 
    output:
        Orthogonal sequence (array)
    '''
    lte_assert(N_PHICH_SF in (2,4), "N_PHICH_SF=%s is out of range! It has to be 4 for normal CP, or 2 for extended CP."%N_PHICH_SF)
    if N_PHICH_SF == 4:
        lte_assert(n_seq_PHICH>=0 and n_seq_PHICH<8, "n_seq_PHICH=%s is out of range!"%n_seq_PHICH)
        result = ( (1,1,1,1), (1,-1,1,-1), (1,1,-1,-1), (1,-1,-1,1), (1j,1j,1j,1j), (1j,-1j,1j,-1j), (1j,1j,-1j,-1j), (1j,-1j,-1j,1j) )[n_seq_PHICH]
    else:
        lte_assert(n_seq_PHICH>=0 and n_seq_PHICH<4, "n_seq_PHICH=%s is out of range!"%n_seq_PHICH)
        result = ( (1,1), (1,-1), (1j,1j), (1j,-1j) )[n_seq_PHICH]
    result = array(result)
    return result
#@+node:michael.20120323224953.1378: *6* 6.9.2 Resource group alignment, layer mapping and precoding
def PHICH_align(d, n_group_PHICH, DL_CP_type, AP_num):
    '''
    input:
        d: modulated and orthogonized symbol sequence
        n_group_PHICH: PHICH group index
        DL_CP_type: downlink CP type, 0 for normal CP, 1 for extended CP
        AP_num: number of antenna ports for CSRS
    output:
        resource group aligned symbol sequence
    '''
    lte_assert(DL_CP_type in (0,1), "DL CP type DL_CP_type=%s is out of range. It has to be either 0 (normal CP) or 1 (extended CP)"%DL_CP_type)
    lte_assert(AP_num in (1,2,4), "Antenna port number AP_num=%s is not correct! It has to be 1, 2, or 4."%AP_num)
    
    # Resource group alignment
    M_symb = len(d)
    if DL_CP_type == 0:
        c = 1
        d_0 = d
    else:
        # extended CP
        c = 2
        d_0 = array( [0.0+0.0*1j] * (c*M_symb) )
        if n_group_PHICH%2 == 0:
            for i in range(M_symb/2):
                d_0[4*i], d_0[4*i+1], d_0[4*i+2], d_0[4*i+3] = d[2*i], d[2*i+1], 0, 0
        else:
            # n_group_PHICH%2 == 1
            for i in range(M_symb/2):
                d_0[4*i], d_0[4*i+1], d_0[4*i+2], d_0[4*i+3] = 0, 0, d[2*i], d[2*i+1]
    return d_0

def PHICH_layer_map_and_precode(d_0, AP_num, n_group_PHICH, DL_CP_type):
    '''
    input:
        d_0: aligned PHICH symbol sequence
        AP_num: number of antenna ports used for CSRS
        n_group_PHICH: PHICH group number
        DL_CP_type: downlink CP type, 0 for normal CP, 1 for extended CP
    output:
        layer mapped matrix
    '''
    lte_assert(AP_num in (1,2,4), "Number of antenna ports for CSRS AP_num=%s is out of number!"%AP_num)
    if AP_num == 1:
        result = layer_map_single_antenna_port(d_0)
        result = precode_single_antenna(result)
    elif AP_num == 2:
        result = layer_map_transmit_diversity( array([d_0]), 2 )
        result = precode_transmit_diversity( result, AP_num )
    else:
        # AP_num == 4
        M_0_symb = len(d_0)
        y = array([ [0.0+0.0*1j]*M_0_symb, [0.0+0.0*1j]*M_0_symb, [0.0+0.0*1j]*M_0_symb, [0.0+0.0*1j]*M_0_symb ])
        tmp_x = array([real(x[0][i]), real(x[1][i]), real(x[2][i]), real(x[3][i]), imag(x[0][i]), imag(x[1][i]), imag(x[2][i]), imag(x[3][i])])
        for i in range(3):
            if (DL_CP_type==0 and (i+n_group_PHICH)%2==0) or (DL_CP_type==1 and (i+n_group_PHICH/2)%2==0):
                y[0][4*i] = 1/sqrt(2) * sum(array([1,0,0,0,1j,0,0,0])) * tmp_x
                y[1][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[2][4*i] = 1/sqrt(2) * sum(array([0,-1,0,0,0,1j,0,0])) * tmp_x
                y[3][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[0][4*i+1] = 1/sqrt(2) * sum(array([0,1,0,0,0,1j,0,0])) * tmp_x
                y[1][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[2][4*i+1] = 1/sqrt(2) * sum(array([1,0,0,0,-1j,0,0,0])) * tmp_x
                y[3][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[0][4*i+2] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,1j,0])) * tmp_x
                y[1][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[2][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,-1,0,0,0,1j])) * tmp_x
                y[3][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[0][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,1,0,0,0,1j])) * tmp_x
                y[1][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[2][4*i+3] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,-1j,0])) * tmp_x
                y[3][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
            else:
                # (i+n_group_PHICH)%2 == 1
                y[0][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[1][4*i] = 1/sqrt(2) * sum(array([1,0,0,0,1j,0,0,0])) * tmp_x
                y[2][4*i] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[3][4*i] = 1/sqrt(2) * sum(array([0,-1,0,0,0,1j,0,0])) * tmp_x
                y[0][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[1][4*i+1] = 1/sqrt(2) * sum(array([0,1,0,0,0,1j,0,0])) * tmp_x
                y[2][4*i+1] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[3][4*i+1] = 1/sqrt(2) * sum(array([1,0,0,0,-1j,0,0,0])) * tmp_x
                y[0][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[1][4*i+2] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,1j,0])) * tmp_x
                y[2][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[3][4*i+2] = 1/sqrt(2) * sum(array([0,0,0,-1,0,0,0,1j])) * tmp_x
                y[0][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[1][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,1,0,0,0,1j])) * tmp_x
                y[2][4*i+3] = 1/sqrt(2) * sum(array([0,0,0,0,0,0,0,0])) * tmp_x
                y[3][4*i+3] = 1/sqrt(2) * sum(array([0,0,1,0,0,0,-1j,0])) * tmp_x
        result = y
    return result
#@+node:michael.20120323224953.1379: *6* 6.9.3 Mapping to resource elements
def get_PHICH_mapping_units(LTE_mode, N_maxDL_RB, N_DL_RB, N_RB_sc, CSRS_AP_num, m_i, N_group_PHICH, DL_CP_type, is_MBSFN, PHICH_duration, subframe, n_s, N_cell_ID, N_DL_symb):
    '''
    input:
        LTE_mode: 'FDD' or 'TDD'
        N_maxDL_RB
        N_DL_RB
        N_RB_sc
        CSRS_AP_num: number of antenna ports used for CSRS
        m_i: times of PHICH group numbers in TDD mode
        N_group_PHICH: number of PHICH groups
        DL_CP_type: downlink CP type, 0 for normal, 1 for extended
        is_MBSFN: boolean, whether this subframe is a MBSFN subframe
        PHICH_duration: 0 for normal duration, 1 for extended duration
        subframe: subframe index
        n_s: slot index
        N_cell_ID: cell ID
        N_DL_symb: downlink symbol number in one slot.
    output:
        a list of resources for all PHICH mapping units, e.g. output[0] is a tuple of 3 REGs, which in turn is a tuple of 4 REs like (k,l), for PHICH mapping unit m_=0. E.g. output[0][0] will be the 1st REG for mapping unit 0.
    '''
    # get the total number of PHICH mapping units
    # M_ is the total number of PHICH mapping units in this slot, i.e. for this PHICH PHY channel
    if DL_CP_type == 0:
        if LTE_mode == 'FDD':
            M_ = N_group_PHICH
        else:
            # LTE_mode == 'TDD'
            M_ = m_i * N_group_PHICH
    else:
        # DL_CP_type == 1: extended CP
        if LTE_mode == 'FDD':
            M_ = N_group_PHICH/2
        else:
            # LTE_mode == 'TDD'
            M_ = (m_i * N_group_PHICH)/2
    
    result = [ [0,0,0] ] * M_
    
    # get total possible symbols to use
    # L_ is the total number of symbols PHICH could use
    if PHICH_duration == 0:
        # normal duration
        L_ = 1
    else:
        # PHICH_duration == 1 for extended duration
        if is_MBSFN:
            L_ = 2
        else:
            # non-MBSFN
            if LTE_mode=='TDD' and subframe in (1,6):
                L_ = 2
            else:
                L_ = 3
    
    # N_L_ is a list of number of REGs not assigned to PCFICH in symbol l_, e.g. N_L_[0] is the number of REGs in symbol 0 that are not assigned to PCFICH
    N_L_ = [0] * L_
    for l in range(L_):
        N_L_[l] = get_REG_number_in_symbol(N_DL_RB, N_RB_sc, l, CSRS_AP_num, DL_CP_type) - get_PCFICH_REG_num_in_symbol(l, True)
        # Assumption here: when there's PHICH, there always is PDSCH in this subframe too.
    
    # a list of all CSRS REs in this slot
    CSRS_REs = tuple()
    for antenna_port in range(CSRS_AP_num):
        CSRS_REs += get_CSRS_REs_in_slot(n_s, antenna_port, N_cell_ID, N_maxDL_RB, N_DL_RB, N_DL_symb)
    # REG_list_in_symbol contains only REGs that are not used by PCFICH, e.g. REG_list_in_symbol[0][0] is the first REG not used by PCFICH in symbol 0.
    REG_list_in_symbol = [0] * L_
    for l in range(L_):
        pcfich_REs = get_PCFICH_REs_in_symbol(l, True, N_cell_ID, N_DL_RB, N_RB_sc, CSRS_REs, CSRS_AP_num, CP_DL_type)
        if len(pcfich_REs) == 0:
            # there's no PCFICH in this symbol
            REG_list_in_symbol[l] = get_REG_in_symbol(N_DL_RB, N_RB_sc, l, CSRS_AP_num, DL_CP_type, CSRS_REs)
        else:
            # there IS PCFICH in this symbol
            available_reg_list = list()
            for reg in get_REG_in_symbol(N_DL_RB, N_RB_sc, l, CSRS_AP_num, DL_CP_type, CSRS_REs):
                if reg[0] not in pcfich_REs:
                    available_reg_list.append(reg)
            REG_list_in_symbol[l] = tuple(available_reg_list)
    
    # start of calculating mapping unit m_
    m_ = 0
    for i in range(3):
        # decide symbol index
        if PHICH_duration == 0:
            l_ = 0
        else:
            # PHICH_duration == 1, extended duration
            if is_MBSFN:
                l_ = (m_/2+i+1)%2
            elif LTE_mode=='TDD' and subframe in (1,6):
                l_ = (m_/2+i+1)%2
            else:
                l_ = i
        # decide subcarrier index
        if (PHICH_duration==1 and is_MBSFN) or (LTE_mode=='TDD' and subframe in (1,6) and PHICH_duration==1):
            n_ = ( N_cell_ID*N_L_[l_]/N_L_[1]+m_+(i*N_L_[l_]/3) )%N_L_[l_]
        else:
            n_ = ( N_cell_ID*N_L_[l_]/N_L_[0]+m_+(i*N_L_[l_]/3) )%N_L_[l_]
        #print m_, i, l_, n_
        #print result
        #print result[m_][i]
        result[m_][i] = REG_list_in_symbol[l_][n_]
        
    return result

def get_PHICH_REs(LTE_mode, N_maxDL_RB, N_DL_RB, N_RB_sc, CSRS_AP_num, m_i, N_group_PHICH, DL_CP_type, is_MBSFN, PHICH_duration, subframe, n_s, N_cell_ID, N_DL_symb):
    '''
    input:
        LTE_mode: 'FDD' or 'TDD'
        N_maxDL_RB
        N_DL_RB
        N_RB_sc
        CSRS_AP_num: number of antenna ports used for CSRS
        m_i: times of PHICH group numbers in TDD mode
        N_group_PHICH: number of PHICH groups
        DL_CP_type: downlink CP type, 0 for normal, 1 for extended
        is_MBSFN: boolean, whether this subframe is a MBSFN subframe
        PHICH_duration: 0 for normal duration, 1 for extended duration
        subframe: subframe index
        n_s: slot index
        N_cell_ID: cell ID
        N_DL_symb: downlink symbol number in one slot.
    output:
        a list of all REs for PHICH in this subframe, e.g. output[0] is the 1st RE.
    '''
    PHICH_mapping_units = get_PHICH_mapping_units(LTE_mode, N_maxDL_RB, N_DL_RB, N_RB_sc, CSRS_AP_num, m_i, N_group_PHICH, DL_CP_type, is_MBSFN, PHICH_duration, subframe, n_s, N_cell_ID, N_DL_symb)
    result = list()
    for mapping_unit in PHICH_mapping_units:
        for reg in mapping_unit:
            for re in reg:
                result.append( re )
    return tuple(result)
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
    input:
        n_s: slot index
        antenna_port: antenna port for CSRS
        N_cell_ID: cell ID
        N_maxDL_RB: 110 for 20MHz configured by higher layer
        N_DL_RB: PHY number of downlink RB
        N_DL_symb: maximum 110 for 20MHz
    output:
        a tuple of all CSRS REs for the given antenna port in the given slot
    
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
        for m in range(2*N_DL_RB):
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
#@+node:michael.20120323224953.1796: *5* 5.3.5 HARQ indicator (HI)
def channel_code_HI(HI):
    '''
    input:
        HI: HARQ indicator, 1 bit, 0 for NACK, 1 for ACK
    output:
        b: 3-bit sequence
    '''
    lte_assert(HI in (0,1), "HI=%s is out of range, it has to be 0 for NACK or 1 for ACK!"%HI)
    return ( (0,0,0), (1,1,1) )[HI]
    
#@-others
#@-others
#@+node:Michael.20120319125504.1481: *3* Error handling
class LteException(Exception):
    
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return repr(self.value)
    
def lte_assert(condition, error_string):
    if not condition:
        raise LteException(error_string)

def lte_warn(warn_string):
    print warn_string
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
