
## Files:

### \_I{1, 2, 3}.csv

Flows refering the scenarios 1, 2 and 3, where there was botnets being executed.

### \_L.csv

Flows with no botnet infection.


## Directories:

### MC: Multimedia Centre

Multimedia centre is a device that consumes streams of video, i.e. Google Chromecast

### ST: Surveillance Camera with additional traffic

As its name says, it refers to traffic due cameras. It includes traffic due access with a webpage, SSH or Telnet and traffic due video streaming.

### SC: Surveillance Camera

As the one above, but not including webpage access, SSH or Telnet.

## Subirectories:

Each of the above directories has the following subdirectories.

### HostData: 
Contains the .csv's files with computers consumption data. They are not used on this work.

### NetFlow: 
The .csv's files with the mentioned features.

### PCAP:
 The PCAP files.

## Features

### From features_netflow.txt
  	
Label:    1: attack; 0: normal  	
ts:       Start Time - first seen  	
te:       End Time - last seen  	
td:       Duration  	
sa:       Source Address  	
da:       Destination Address  	
sp:       Source Port  	
dp:       Destination Port  	
pr:       Protocol  	
flg:      TCP Flags  	
fwd:      Forwarding Status  	
stos:     Src Tos  	
ipkt:     Input Packets  	
ibyt:     Input Bytes  	
opkt:     Output Packets  	
obyt:     Output Bytes  	
in:       Input Interface num  	
out:      Output Interface num  	
sas:      Source AS  	
das:      Destination AS  	
smk:      Src mask  	
dmk:      Dst mask  	
dtos:     Dst Tos  	
dir:      Direction: ingress, egress  	
nh:       Next-hop IP Address  	
nhb:      BGP Next-hop IP Address  	
svln:     Src vlan label  	
dvln:     Dst vlan label  	
idmc:     Input Dst Mac Addr  	
osmc:     Output Src Mac Addr  	
mpls1:    MPLS label 1  	
mpls2:    MPLS label 2  	
mpls3:    MPLS label 3  	
mpls4:    MPLS label 4  	
mpls5:    MPLS label 5  	
mpls6:    MPLS label 6  	
mpls7:    MPLS label 7  	
mpls8:    MPLS label 8  	
mpls9:    MPLS label 9  	
mpls10:   MPLS label 10  	
cl:       Client latency  	
sl:       Server latency  	
al:       Application latency  	
ra:       Router IP Address  	
eng:      Engine Type/ID  	
exid:     Exit number  	
tr:       Time the flow was received by the collector    	
