create database moviedb;

use moviedb;

create table movietbl
	(movie_id		int,
    movie_title 	varchar(30),
    movie_director	varchar(20),
    movie_star		varchar(20),
    movie_script	LONGTEXT,
    movie_film		LONGBLOB)
default charset=utf8mb4;

insert into movietbl values(1,'쉰들러 리스트','스필버그','리암 니스',
load_file('F:\MySQL\movies\Schindler.txt'),
load_file('F:\MySQL\movies\Schindler.mp4'));

-- 데이터가 null로 되어 있음, 기본 업로드 데이터양이 작아서, 보안 파일 설정이 되어 있어서 --
select * from movietbl;

show variables like 'max_allowed_packet';

show variables like 'secure_file_priv';

-- 시스템 설정 변경하기 -- 


use moviedb;
truncate movietbl; -- 기존 행 모두 제거

insert into movietbl values(1,'쉰들러 리스트','스필버그','리암 니스',
load_file("F:\MySQL\movies\Schindler.txt"),
load_file("F:\MySQL\movies\Schindler.mp4"));

insert into movietbl values(2,'쇼생크 탈출','프랭크 다라본트','팀 로빈스',
load_file("F:\MySQL\movies\Shawshank.txt"),
load_file("F:\MySQL\movies\Shawshank.mp4"));

insert into movietbl values(3,'라스트 모히칸','마이클 만','다니엘 데이 루이스',
load_file("F:\MySQL\movies\Mohican.txt"),
load_file("F:\MySQL\movies\Mohican.mp4"));

select * from movietbl;

select movie_script from movietbl where movie_id=1
into outfile "F:\Schindler_out.txt"
lines terminated by '\\n';

select movie_script from movietbl where movie_id=3
into dumpfile "F:\Schindler_out.mp4"



