#!/usr/bin/python3
#---lujy 2020.3---

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

def convert(path, fileType, dstEnc='UTF-8'):
    import chardet
    suffixDict = {
        'c':(r'.cc', r'.h', r'.cpp', r'.c'),
        'py':(r'.py'),
        'm':(r'.m'),
        'txt':(r'txt')
    }
    flist = getFile(path, suffixDict[fileType])
    for file in flist:
        with open(file, 'rb') as f:
            data = f.read()
            codeType = chardet.detect(data)['encoding']
            _convert(file, codeType, dstEnc)


def showHelp():
    print('Usage: cvt -o [dst_encode] -t [source type] --path=[source path]')
    print('source type:')
    print('c: c/c++ files')
    print('py：python files')
    print('m: matlab files')
    print('txt: text files')


if __name__ == '__main__':
    import chardet
    import sys
    import getopt
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "-h-o-t:-p:", ['help', 'path='])
    except getopt.GetoptError as err:
        print(str(err))
        showHelp()
        sys.exit()
    
    path=''
    dstEnc = 'UTF-8'
    fileType=''
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            showHelp()
            sys.exit()
        elif opt in ('-o'):
            dstEnc = arg
        elif opt in ('-t'):
            fileType = arg
        elif opt in ('-p', '--path'):
            path = arg
        else:
            assert False, 'Unhandled Option'
            
    if path == '' or fileType == '':
        showHelp()
        sys.exit()
    
    convert(path, fileType, dstEnc)
    
    sys.exit()