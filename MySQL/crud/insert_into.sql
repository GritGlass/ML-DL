-- usertbl에서 데이터3개 가져옴 => membertbl --
use sqldb;
create table membertbl (select userID, name, addr from usertbl limit 3);
alter table membertbl
add CONSTRAINT pk_membertbl PRIMARY KEY (userID); -- pk를 지정함
select * from membertbl

insert into membertbl values('BBK','비비코','미국'); -- 에러 발생 => PK중복 'BBK'
insert into membertbl values('SJH','서장훈','서울');
insert into membertbl values('HJY','현주엽','경기');
select * from membertbl; -- 위 데이터 3개 모두 추가 안됨

insert ignore into membertbl values('BBK','비비코','미국'); -- 에러 발생 => 무시
insert ignore into membertbl values('SJH','서장훈','서울');
insert ignore into membertbl values('HJY','현주엽','경기');
select * from membertbl; -- pk 중복된 데이터 제외한 나머지 2개 데이터 추가 됨

insert into membertbl values('BBK','비비코','미국') 
	on DUPLICATE KEY UPDATE name='비비코', addr='미국'; -- PK중복시 내용 변경
insert into membertbl values('DJM','동짜몽','일본')
	on DUPLICATE KEY UPDATE name='동짜몽', addr='일본';
select * from membertbl; -- 비비코 내용 변경, 동짜몽 데이터 추가 됨