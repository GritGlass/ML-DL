-- 데이터 삽입 --
insert into membertbl values('Figure','연아','경기도 군포시 당정동');
select * from membertbl;

-- 데이터 수정 --
update membertbl set memberaddress = '서울 강남구 역삼동' where membername='연아';
select * from membertbl;

-- 데이터 삭제 --
delete from membertbl where membername='연아';
select * from membertbl;

-- 탈퇴 회원 테이블 생성 --
create table deletemembertbl(
	memberid char(8),
    membername char(5),
    memberaddress char(20),
    deletedate date
    );
    
-- 트리거 생성 : 회원 탈퇴시, 탈퇴 회원 정보 -> 탈퇴 회원 테이블 --
DELIMITER //
create trigger trg_deletedmembertbl 
	after delete
    on membertbl
    for each row
begin
	insert into deletemembertbl
		values (old.memberid, old.membername, old.memberaddress, CURDATE());
end //
DELIMITER ;


SELECT * FROM MEMBERTBL;

DELETE FROM membertbl WHERE memberName= '당탕이' ;

Select * from membertbl;

select * from deletemembertbl;
