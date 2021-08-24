import pandas as pd

def get_data_chunk_qicpic_txt_output(fname, ref_tag, size_chunk, incr_to_start = 1):
    df = pd.read_csv(fname, encoding = "ISO-8859-1", header=None)
    start_index_dens_txt = df.index[df[0] == ref_tag]
    start_index = start_index_dens_txt + incr_to_start
    df_clear = [df.iloc[start_index[0]: start_index[0]+size_chunk, 0:2]
                .set_index(pd.Index(range(size_chunk)))
                   for i in range(len(start_index))]
    for i in range(len(df_clear)):
        df_clear[i][0] = pd.to_numeric(df_clear[i][0])
        df_clear[i][1] = pd.to_numeric(df_clear[i][1])
    return df_clear


if __name__ == "__main__":
    df = pd.read_csv('QICPICa_TC2.csv', encoding = "ISO-8859-1", header=None)
    SIZE_PSD_IN_TXT = 31
    incr_to_start = 6    
    tag = 'Distribution density (log.)'
    args = ('QICPICa_TC2.csv', tag, SIZE_PSD_IN_TXT, incr_to_start)
    df_psd_log = get_data_chunk_qicpic_txt_output(*args)
    print('---------- Variable: {} --------- '.format(tag))
    print('Size = {}'.format(len(df_psd_log)))

    tag = 'distribution density (lin.)'
    args = ('QICPICa_TC2.csv', tag, SIZE_PSD_IN_TXT, incr_to_start)
    df_psd_log = get_data_chunk_qicpic_txt_output(*args)
    print('---------- Variable: {} --------- '.format(tag))
    print('Size = {}'.format(len(df_psd_log)))    
    pass