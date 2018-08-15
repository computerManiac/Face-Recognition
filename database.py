import sqlite3
import os

DEFAULT_PATH = "database/records.sqlite3"

def db_connect(file_path=DEFAULT_PATH):
	conn = sqlite3.connect(file_path)
	return conn

def db_create(c_obj):

	cursor = c_obj.cursor()
	create_sql = '''
		CREATE TABLE users(
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		embeddings TEXT NOT NULL
		)
		'''
	cursor.execute(create_sql)
	c_obj.commit()


def db_insert(c_obj,name,emb):
	
	cursor = c_obj.cursor()

	insert_sql = '''
		INSERT INTO users(name,embeddings) VALUES (?,?)
	'''
	cursor.execute(insert_sql,(name,emb))
	c_obj.commit()

def db_getData(c_obj):

	cursor = c_obj.cursor()
	cursor.execute('''
		SELECT name,embeddings FROM users
		''')
	rows = cursor.fetchall()

	data =[]

	for row in rows:
		temp = []

		temp.append(row[0])
		temp.append(row[1])

		data.append(temp)

	return data

def get_names(c_obj):
	cursor = c_obj.cursor()
	cursor.execute('''
		SELECT DISTINCT name FROM users
		''')
	rows = cursor.fetchall()
	names = []
	for r in rows:
		names.append(r[0])
	return names