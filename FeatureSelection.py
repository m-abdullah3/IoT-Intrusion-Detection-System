import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

#Loading the dataset as a pandas frame
dataFrame= pd.read_csv("preprocessed_CIC-NEW.csv")

#printing the shape on the dataset
print ("shape = ",dataFrame.shape)

#Listing the features to be consdiered (all in our case)
factors=['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
       'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
       'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
       'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
       'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
       'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'label']

#Finding the correaltions between the features
correlations=dataFrame[factors].corr()
#Creating a figure with multiple subplots to accomdate the heat map
plt.subplots(figsize=(50, 50))

# Creating the heatmap using seaborn
sns.heatmap(correlations, square=True, vmin=-1, vmax=1, annot=False, cmap='BrBG');
#Showing the plotted heatmap
plt.show()

