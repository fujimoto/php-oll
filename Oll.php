<?php
/**
 * Online Learning for PHP (just an experimental implementation)
 *
 * PHP versions 5
 *
 * LICENSE: This source file is subject to version 3.0 of the PHP license
 * that is available through the world-wide-web at the following URI:
 * http://www.php.net/license/3_0.txt. If you did not receive a copy of
 * the PHP License and are unable to obtain it through the web, please
 * send a note to license@php.net so we can mail you a copy immediately.
 *
 * @author Masaki Fujimoto <fujimoto@php.net>
 * @license http://www.php.net/license/3_0.txt PHP License 3.0
 */
// requires mbstring :(
mb_internal_encoding('UTF-8');

abstract class Oll_Storage {
	abstract public function __construct();
	abstract public function open();
	abstract public function get($key);
	abstract public function set($key, $value);
}

class Oll_Storage_Sqlite extends Oll_Storage {
	const	default_path = "/var/tmp/oll.db";
	private	$path;
	private	$db = null;

	public function __construct($path = null) {
		$this->path = $path === null ? self::default_path : $path;
	}

	public function open() {
		if ($this->db !== null) {
			return true;
		}

		$skip_create = file_exists($this->path);
		$this->db = new SQLiteDatabase($this->path);

		if ($skip_create === false) {
			$sql = "CREATE TABLE oll (k VARBINARY(16), v DOUBLE, PRIMARY KEY (k))";
			$this->db->query($sql);
		}

		return true;
	}

	public function get($key) {
		$sql = sprintf("SELECT v FROM oll WHERE k='%s'", sqlite_escape_string($key));
		$r = $this->db->query($sql);
		if ($r->numRows() <= 0) {
			return null;
		}
		$row = $r->fetch();

		return $row['v'];
	}

	public function set($key, $value) {
		$sql = sprintf("REPLACE INTO oll (k, v) VALUES ('%s', '%s')", sqlite_escape_string($key), sqlite_escape_string($value));
		$r = $this->db->query($sql);

		return true;
	}
}

abstract class Oll {
	protected $storage;

	abstract public function train($x, $y);
	abstract public function test($x);

	public function __construct($storage) {
		$this->storage = $storage;
	}

	public static function createInstance($algorithm, $storage) {
		$klass = sprintf('Oll_%s', ucfirst($algorithm));
		if (class_exists($klass) === false) {
			throw new Exception("$algorithm is not yet supported");
		}
		$object = new $klass($storage);

		return $object;
	}

	public static function makeVector($content) {
		$x = array();
		if ($content == "") {
			return $x;
		}

		// applying filters will increase accuracy... (morphological analysis, and some more heuristics)
		// and better make this pluggable (todo)
		$content = str_replace(array("\r", "\n"), '', $content);

		// 2-6 grams
		$len = mb_strlen($content);
		for ($i = 0; $i < $len; ++$i) {
			for ($j = 2; $j <= 6; ++$j) {
				if ($i + $j - 1 < $len) {
					$key = mb_substr($content, $i, $j);
					if (isset($x[$key]) === false) {
						$x[$key] = 0;
					}
					$x[$key] += 1;
				}
			}
		}
		if (count($x) == 0) {
			return $x;
		}

		return $x;
	}
}

class Oll_Perceptron extends Oll {
	const		bias_key = "_oll_perceptron_bias_";
	private		$w = array();
	private		$bias = null;
	protected	$update_score_threshold = 0.0;

	public function __construct($storage) {
		parent::__construct($storage);

		$this->bias = $this->_get(self::bias_key);
		if ($this->bias === null) {
			$this->bias = 0.0;
		}
	}

	public function train($x, $y) {
		$score = $y * $this->_getMargin($x);
		if ($score <= $this->update_score_threshold) {
			$this->_update($x, $y * $this->_normalize($x, $score));
		}

		return true;
	}

	public function test($x) {
		return $this->_getMargin($x);
	}

	protected function _normalize($x) {
		return 1;
	}

	private function _getMargin($x) {
		$bias = $this->bias;
		foreach ($x as $k => $v) {
			$bias += $this->_get($k) * $v;
		}

		return $bias;
	}

	private function _update($x, $y) {
		foreach ($x as $k => $v) {
			$this->_set($k, $this->_get($k) + $y * $v);
		}

		$this->bias *= $y;
		$this->_set(self::bias_key, $this->bias);
	}

	private function _get($k) {
		if (isset($this->w[$k]) === false) {
			$v = $this->storage->get($k);
			if ($v === null) {
				$v = 0.0;
			}
			$this->w[$k] = $v;
		}

		return $this->w[$k];
	}

	private function _set($k, $v) {
		$this->w[$k] = $v;
		$this->storage->set($k, $v);
	}
}

class Oll_Perceptron_Aggressive extends Oll_Perceptron {
	protected	$update_score_threshold = 1.0;

	protected function _normalize($x, $score) {
		$bias = 1.0;
		foreach ($x as $k => $v) {
			$bias += $v * $v;
		}
		return $bias;
	}
}

class Oll_Perceptron_Aggressive1 extends Oll_Perceptron {
	const		delta = 1.0;
	protected	$update_score_threshold = 1.0;

	protected function _normalize($x, $score) {
		$bias = 1.0;
		foreach ($x as $k => $v) {
			$bias += $v * $v;
		}
		return min(self::delta, (1.0 - score) / $bias);
	}
}

class Oll_Perceptron_Aggressive2 extends Oll_Perceptron {
	const		delta = 1.0;
	protected	$update_score_threshold = 1.0;

	protected function _normalize($x, $score) {
		$bias = 1.0;
		foreach ($x as $k => $v) {
			$bias += $v * $v;
		}
		return (1.0 - $score) / ($bias + 1.0 / 2.0 / self::delta);
	}
}

class Oll_NaiveBayse extends Oll {
	const		total_key = "_oll_naivebayse_totla_key_";
	private		$w = array();
	private		$total = array(0.0000001, 0.0000001);

	public function __construct($storage) {
		parent::__construct($storage);
		$d = $storage->get(self::total_key . "0");
		if ($d !== null) {
			$this->total[0] = $d;
		}
		$d = $storage->get(self::total_key . "1");
		if ($d !== null) {
			$this->total[1] = $d;
		}
	}

	public function train($x, $y) {
		if ($y > 0) {
			foreach ($x as $k => $v) {
				$t = $this->_get($k);
				$this->_set($k, array($t[0] + intval($v), $t[1]));
				$this->total[0] += intval($v);
			}
			$this->storage->set(self::total_key . "0", $this->total[0]);
		} else {
			foreach ($x as $k => $v) {
				$t = $this->_get($k);
				$this->_set($k, array($t[0], $t[1] + intval($v)));
				$this->total[1] += intval($v);
			}
			$this->storage->set(self::total_key . "1", $this->total[1]);
		}
	}

	public function test($x) {
		$pp = 0.0;
		$np = 0.0;
		list($pt, $nt) = $this->total;
		foreach ($x as $key => $value) {
			list($p, $n) = $this->_get($k);
			$pp += log(floatval($p) / $pt) * intval($v);
			$np += log(floatval($n) / $nt) * intval($v);
		}
		$pp += log(floatval($pt) / ($pt + $nt));
		$np += log(floatval($nt) / ($pt + $nt));

		return ($pp > $np) ? 1 : -1;
	}

	private function _get($k) {
		if (isset($this->w[$k])) {
			return $this->w[$k];
		}

		$d0 = $this->storage->get($key . "0");
		$d1 = $this->storage->get($key . "1");
		if ($d0 !== null && $d1 !== null) {
			$this->w[$k] = array($d0, $d1);
		} else {
			$this->w[$k] = array(0.0000001, 0.0000001);
		}

		return $this->w[$k];
	}

	private function _set($k, $v) {
		$this->w[$k] = $v;
		$this->storage->set($k . "0", $v[0]);
		$this->storage->set($k . "1", $v[1]);
	}
}
