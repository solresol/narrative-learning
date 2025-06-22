import unittest
import psycopg2

class TestPostgresConnection(unittest.TestCase):
    def test_can_query_database(self):
        conn = psycopg2.connect(dbname='narrative', user='root')
        cur = conn.cursor()
        cur.execute('SELECT 1')
        self.assertEqual(cur.fetchone()[0], 1)
        cur.execute('CREATE TEMP TABLE temp_test (id INTEGER)')
        cur.execute('INSERT INTO temp_test (id) VALUES (42)')
        cur.execute('SELECT id FROM temp_test')
        self.assertEqual(cur.fetchone()[0], 42)
        conn.close()

if __name__ == '__main__':
    unittest.main()
