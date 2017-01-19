from datetime import datetime
debug_level = 'dev'# production, log


def print_log(data):
	if True: # debug_level
		try:
			if type(data) is list or type(data) is dict:
				print '[' + str(datetime.now()) + ']'
				for d in data:
					print str(d)
			else:
				print '[' + str(datetime.now()) + '] ' + str(data)
		except Exception as TypeError:
			print 'ATTENTION [' + str(datetime.now()) + '] ' + 'Printing error in log.print_log'


def write_log(data,filename):
	return

def update_log(data, table, database):
	return






