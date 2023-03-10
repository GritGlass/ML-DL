select * from membertbl where memberName='당탕이';
select * from producttbl where productid='냉장고';

DELIMITER //
CREATE PROCEDURE myProc()
BEGIN
	select * from membertbl where memberName='당탕이';
	select * from producttbl where productid='냉장고';
END //
DELIMITER ;

CALL myProc()

