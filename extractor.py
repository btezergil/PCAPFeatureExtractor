import pyshark
import datetime, argparse

PCAP_PATH = '/home/btezergil/Desktop/research488/hangouts/'

class Features:
    def __init__(self, startTime, interval):
        self.startTime = startTime
        self.interval = interval

        self.httpExists = 0
        self.DTLSExists = 0
        self.ICMPExists = 0

        self.firstSTUNMessageTime = 0
        self.STUNMessageRate = 0
        self.ICMPMessageRate = 0
        self.UDPDatagramRate = 0
        
        self.DNSAtStart = 0
        self.DNSAtEnd = 0

        self.numOfUDPHosts = 0
        self.numOfTLSHosts = 0
        self.numOfSTUNHosts = 0
        self.numOfTCPHosts = 0

        self.TLSMessagePercentage = 0
        self.STUNMessagePercentage = 0
        self.TCPMessagePercentage = 0

def extract_features(file, time_interval):
    # open the capture file and set up the parameters for the sliding time window
    td = datetime.timedelta(seconds = time_interval) # set the timedelta of the interval for window sliding
    cap = pyshark.FileCapture(file)
    startTime = cap[0].sniff_time
    endTime = cap[0].sniff_time + td

    #print(startTime.strftime("%A, %d. %B %Y %I:%M.%S%p"))
    #print(endTime.strftime("%A, %d. %B %Y %I:%M.%S%p"))

    # prepare the feature dictionary and list of the features at every time window
    # TODO: HIS WILL BE DONE AFTER CHECKING OUT HMM LIBRARIES
    # THE FEATURE FORMAT IS DECIDED BY THOSE LIBRARIES AND FEATURES WILL BE PREPARE ACCORDINGLY

    startCount = 1
    endCount = 0

    # start processing the packets 
    while True:
        # get the next packet from capture
        try:
            a = cap.next()
        except StopIteration as e:
            break

        t = a.sniff_time
        
        # check the time window
        if endTime < t:
            # finalize the feature list, prepare a new list, update startTime and endTime
            startTime = endTime
            endTime = startTime + td
            # TODO: finalize the feature list
            print("Packets {}-{} processed".format(startCount, endCount))
            startCount = a.frame_info.number

        endCount += 1


        #print(a.sniff_time.strftime("%A, %d. %B %Y %I:%M.%S%p"))
    print("Extraction finished.")


    return 

parser = argparse.ArgumentParser(description = 'Extract features from given pcap file.')
parser.add_argument("filename", type = str)
parser.add_argument("interval", type = float)
args = parser.parse_args()
print("Extracting from file {} with an interval of {} seconds\n".format(args.filename, args.interval))
extract_features(args.filename, args.interval)