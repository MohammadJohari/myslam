from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'nice': decoder.MyNICE,
    'imap':decoder.MLP
}