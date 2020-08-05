import smtplib

sender = 'giserman@gta.ufrj.br'
receivers = ['ernesto@gta.ufrj.br', 'kaylani@gta.ufrj.br', 'giserman@gta.ufrj.br']

#To: To us <ernesto@gta.ufrj.br>, <kaylani@gta.ufrj.br>, <giserman@gta.ufrj.br>
message = """From: SBSEG <sbseg@gta.ufrj.br>
To: To us 
Subject: Luiz's code has ended, an output file was created.
Look at the output file!
"""

try:
    smtpObj = smtplib.SMTP('gta.ufrj.br')
    smtpObj.sendmail(sender, receivers, message)
    print ('Successfully sent email.')
except SMTPException:
    print ('Error: unable to send email.')
