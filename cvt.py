#!/usr/bin/python3

def endWith(file, ends):
    for end in ends:
        if file.endswith(end):
            return True
    return False

def getFile(path, end):
    import os
    #all = os.walk(path)
    filelist = []
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)):
            files = getFile(os.path.join(path, filename), end)
            filelist = filelist + files
        elif endWith(filename, end):
            filelist.append(os.path.join(path, filename))
    return filelist

def _convert(file, inEnc, outEnc):
    import codecs
    inEnc = inEnc.upper()
    outEnc = outEnc.upper()
    try:
        print("convert [ " + file + " ].....From " + inEnc + " --> " + outEnc )
        f = codecs.open(file, 'r', inEnc)
        newContent = f.read()
        codecs.open(file, 'w', outEnc).write(newContent)
        #print (f.read())
    except IOError as err:
        print("I/O error: {0}".format(err))


if __name__ == '__main__':
    path = r'C:\Users\lujy\Desktop\utf8'
    flist = getFile(path, [r'.txt'])
    #path = r'y:\ms'
    #flist = getFile(path, [r'.cc', r'.h', r'.cpp', r'.c'])
    print(flist)
    
    import chardet
    for file in flist:
        with open(file, 'rb') as f:
            data = f.read()
            codeType = chardet.detect(data)['encoding']
            _convert(file, codeType, 'UTF-8')
            