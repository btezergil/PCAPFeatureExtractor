import pyshark
import datetime, argparse

PCAP_PATH = '/home/btezergil/Desktop/research488/hangouts/'

# for atStart and atEnd variables, we choose a fixed time window
START_WINDOW = 5
END_WINDOW = 5

class Features:
    def __init__(self, startTime, interval):
        self.startTime = startTime
        self.interval = interval
        
        # exists parameters can only be 0 or 1
        # DONE
        self.HTTPExists = 0
        self.DTLSExists = 0
        self.ICMPExists = 0

        # time parameter 1 if detect time is less than START_WINDOW, 0 otherwise
        # rates are 0 if less than 0.1, 1 if less than 0.2, 2 otherwise
        # DONE
        self.firstSTUNMessageTime = 0
        self.STUNMessageRate = 0.0
        self.ICMPMessageRate = 0.0
        self.UDPDatagramRate = 0.0
        
        # 0 or 1
        # DONE
        self.DNSAtStart = 0
        self.DNSAtEnd = 0

        # 1, 11 or 21
        # DONE
        self.numOfUDPHosts = 0
        self.numOfTLSHosts = 0
        self.numOfSTUNHosts = 0
        self.numOfTCPHosts = 0

        # below 1, below 4, above 4 perc.
        # DONE
        self.TLSMessagePercentage = 0.0
        self.STUNMessagePercentage = 0.0
        self.TCPMessagePercentage = 0.0

    def getObsArray(self):
        obsarr = []
        obsarr.append(self.HTTPExists)
        obsarr.append(self.DTLSExists)
        obsarr.append(self.ICMPExists)
        obsarr.append(self.firstSTUNMessageTime)
        obsarr.append(self.DNSAtStart)
        obsarr.append(self.DNSAtEnd)
        obsarr.append(self.STUNMessageRate)
        obsarr.append(self.ICMPMessageRate)
        obsarr.append(self.UDPDatagramRate)
        obsarr.append(self.numOfUDPHosts)
        obsarr.append(self.numOfTLSHosts)
        obsarr.append(self.numOfSTUNHosts)
        obsarr.append(self.numOfTCPHosts)
        obsarr.append(self.TLSMessagePercentage)
        obsarr.append(self.STUNMessagePercentage)
        obsarr.append(self.TCPMessagePercentage)
        return obsarr
        

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
    endCount = 1
    startWindow = cap[0].sniff_time + datetime.timedelta(seconds = START_WINDOW)
    endWindow = cap[-1].sniff_time - datetime.timedelta(seconds = END_WINDOW)

    # percentage calculations
    totalCount = 0
    TLSCount = 0
    STUNCount = 0
    TCPCount = 0
    ICMPCount = 0
    UDPCount = 0

    # host number calculations
    UDPHosts = []
    TLSHosts = []
    STUNHosts = []
    TCPHosts = []

    ftrs = Features(startTime, time_interval)
    featureArray = []

    # start processing the packets 
    while True:
        # get the next packet from capture
        try:
            a = cap.next()
        except StopIteration:
            break

        t = a.sniff_time
        totalCount += 1

        # check the protocol of the packet, increment counts or save hosts
        if 'SSL' in a:
            TLSCount += 1
            host = a.ip.host
            if host not in TLSHosts:
                TLSHosts.append(host)
        if 'TCP' in a:
            TCPCount += 1
            host = a.ip.host
            if host not in TCPHosts:
                TCPHosts.append(host)
        if 'UDP' in a:
            UDPCount += 1
            host = a.ip.host
            if host not in UDPHosts:
                UDPHosts.append(host)
        if 'STUN' in a:
            STUNCount += 1
            host = a.ip.host
            if host not in STUNHosts:
                STUNHosts.append(host)
        if 'HTTP' in a:
            ftrs.HTTPExists = 1
        if 'DTLS' in a:
            ftrs.DTLSExists = 1
        if 'ICMP' in a:
            ftrs.ICMPExists = 1
            ICMPCount += 1

        # check atStart and atEnd conditions
        if t < startWindow:
            if 'DNS' in a:
                ftrs.DNSAtStart = 1
            if 'STUN' in a:
                ftrs.firstSTUNMessageTime = 1
        elif t > endWindow:
            if 'DNS' in a:
                ftrs.DNSAtEnd = 1

        
        # check the time window
        if endTime < t:
            # update startTime and endTime
            startTime = endTime
            endTime = startTime + td

            # finalize the feature list, prepare a new list
            # calculate numberOfXHosts fields for HMM
            length = len(UDPHosts)
            if length < 11:
                ftrs.numOfUDPHosts = 0
            elif length < 21:
                ftrs.numOfUDPHosts = 1
            else:
                ftrs.numOfUDPHosts = 2

            length = len(TLSHosts)
            if length < 11:
                ftrs.numOfTLSHosts = 0
            elif length < 21:
                ftrs.numOfTLSHosts = 1
            else:
                ftrs.numOfTLSHosts = 2
            
            length = len(TCPHosts)
            if length < 11:
                ftrs.numOfTCPHosts = 0
            elif length < 21:
                ftrs.numOfTCPHosts = 1
            else:
                ftrs.numOfTCPHosts = 2
            
            length = len(STUNHosts)
            if length < 11:
                ftrs.numOfSTUNHosts = 0
            elif length < 21:
                ftrs.numOfSTUNHosts = 1
            else:
                ftrs.numOfSTUNHosts = 2

            # calculate percentages
            perc = TLSCount / totalCount
            if perc < 1:
                ftrs.TLSMessagePercentage = 0
            elif perc < 4:
                ftrs.TLSMessagePercentage = 1
            else:
                ftrs.TLSMessagePercentage = 2

            perc = STUNCount / totalCount
            if perc < 1:
                ftrs.STUNMessagePercentage = 0
            elif perc < 4:
                ftrs.STUNMessagePercentage = 1
            else:
                ftrs.STUNMessagePercentage = 2

            perc = TCPCount / totalCount
            if perc < 1:
                ftrs.TCPMessagePercentage = 0
            elif perc < 4:
                ftrs.TCPMessagePercentage = 1
            else:
                ftrs.TCPMessagePercentage = 2

            rate = STUNCount / (t - cap[0].sniff_time).total_seconds()
            rate /= 1000
            if rate < 0.1:
                ftrs.STUNMessageRate = 0
            elif rate < 0.2:
                ftrs.STUNMessageRate = 1
            else:
                ftrs.STUNMessageRate = 2

            rate = ICMPCount / (t - cap[0].sniff_time).total_seconds()
            rate /= 1000
            if rate < 0.1:
                ftrs.ICMPMessageRate = 0
            elif rate < 0.2:
                ftrs.ICMPMessageRate = 1
            else:
                ftrs.ICMPMessageRate = 2

            rate = UDPCount / (t - cap[0].sniff_time).total_seconds()
            rate /= 1000
            if rate < 0.1:
                ftrs.UDPDatagramRate = 0
            elif rate < 0.2:
                ftrs.UDPDatagramRate = 1
            else:
                ftrs.UDPDatagramRate = 2

            featureArray.append(ftrs)
            ftrs = Features(startTime, time_interval)

            try:
                # set the features that will be carried on for all iterations
                oldftrs = featureArray[-1]
                if oldftrs.DNSAtStart:
                    ftrs.DNSAtStart = 1
                if oldftrs.DNSAtEnd:
                    ftrs.DNSAtEnd = 1
                if oldftrs.DTLSExists:
                    ftrs.DTLSExists = 1
                if oldftrs.HTTPExists:
                    ftrs.HTTPExists = 1
                if oldftrs.ICMPExists:
                    ftrs.ICMPExists = 1
                if oldftrs.firstSTUNMessageTime:
                    ftrs.firstSTUNMessageTime = 1
            except IndexError:
                pass
            
            print("Packets {}-{} processed".format(startCount, endCount))
            startCount = a.frame_info.number

        endCount += 1


        #print(a.sniff_time.strftime("%A, %d. %B %Y %I:%M.%S%p"))
    print("Extraction finished.")

    return featureArray

def main():
    parser = argparse.ArgumentParser(description = 'Extract features from given pcap file.')
    parser.add_argument("filename", type = str)
    parser.add_argument("interval", type = float)
    args = parser.parse_args()

    print("Extracting from file {} with an interval of {} seconds\n".format(args.filename, args.interval))
    extract_features(args.filename, args.interval)

if __name__ == "__main__":
    main()
