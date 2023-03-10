use sqldb;
create table emptbl (emp char(3), manager char(3), emptel varchar(8));

insert into emptbl values('나사장',null,'0000');
insert into emptbl values('김재무','나사장','2222');
insert into emptbl values('김부장','김재무','2222-1');
insert into emptbl values('이부장','김재무','2222-2');
insert into emptbl values('우대리','이부장','2222-2-1');
insert into emptbl values('지사원','이부장','2222-2-2');
insert into emptbl values('이영업','나사장','1111');
insert into emptbl values('한과장','이영업','1111-1');
insert into emptbl values('최정보','나사장','3333');
insert into emptbl values('윤차장','최정보','3333-1');
insert into emptbl values('이주임','윤차장','3333-1-1');

select A.emp as '부하직원', B.emp as '직속상관', B.emptel as '직속상관연락처'
from emptbl A
inner join emptbl B
on A.manager = B.emp
where A.emp='우대리';